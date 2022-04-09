"""
    대신증권 Cybos API를 활용한 계좌 관리 및 주문집행 모듈(win 32-bit)

    @author: Younghyun Kim
    Created on 2021.06.14
"""
import sys
import traceback
import time
import datetime
from collections import defaultdict
import win32com.client as wc
from pywinauto import application
import locale
import os
import numpy as np
import pandas as pd
import ctypes
import pdb

from merlin.cfg.configs_secret import account_info

locale.setlocale(locale.LC_ALL, 'ko_KR')


class TradingManager:
    """
        Cybos Account Manager
    """
    def __init__(self, acc_info=None, auto_login=True,
                db_name='quantdb',
                conc_table_name='conclusion_series',
                jango_table_name='jango_data',
                db_manager=None, load_stocks=True,
                stocks_selected=None):
        """
            Args:
                acc_info: account information
                auto_login: 자동 로그인 여부(기본값: True)
        """
        if acc_info is None:
            self.acc_info = account_info
        else:
            self.acc_info = acc_info

        account_id = self.acc_info['account_id']
        account_pwd = self.acc_info['account_pwd']
        account_pwdcert = self.acc_info['account_pwdcert']

        self.today = datetime.datetime.now().date()

        self.g_objCpStatus = wc.Dispatch('CpUtil.CpCybos')
        self.g_objCodeMgr = wc.Dispatch('CpUtil.CpCodeMgr')
        self.g_objCpTrade = wc.Dispatch('CpTrade.CpTdUtil')
        self.g_objCpStockCode = wc.Dispatch('CpUtil.CpStockCode')
        self.objOrder = wc.Dispatch('CpTrade.CpTd0311')  # 주문
        self.g_objCur = wc.Dispatch('DsCbo1.StockMst')

        self.account_id = account_id
        self.account_pwd = account_pwd
        self.account_pwdcert = account_pwdcert

        self.db_name = db_name
        self.conc_table_name = conc_table_name
        self.jango_table_name = jango_table_name

        self.auto_login = auto_login

        if auto_login:
            self.connect(account_id, account_pwd, account_pwdcert)

            if self.init_plus_check() == False:
                exit()

        # 6033 잔고 object
        self.obj6033 = Cp6033()
        self.jangoData = {}
        self.yesu = None  # D+2 예상예수금

        # 계좌번호
        self.acc = self.g_objCpTrade.AccountNumber[0]
        self.accFlag =\
            self.g_objCpTrade.GoodsList(self.acc, 1)  # 주식상품

        self.isSB = False
        self.objCur = {}

        # 현재가 정보
        self.curDatas = {}
        self.objRPCur = CpRPCurrentPrice()

        # 주문 정보
        self.order_data = {}
        self.daily_conc_history = []

        # 실시간 주문 체결
        self.objConclusion = CpPBConclusion()

        # 주문 취소
        self.objCancelOrder = CpRPOrder(self)

        # 미체결 조회 서비스
        self.obj5339 = Cp5339(self)

        # 잔고 컬럼명
        self.jango_columns = ['trade_date',  # 영업일
                              'stock_code',  # 종목코드
                              'stock_name',  # 종목명
                              'stock_amount',  # 잔고수량
                              'stock_amount_available',  # 매도가능
                              'book_value',  # 장부가
                              'purchase_value_book_based',
                              # 매입금액(장부가 기준)
                              'purchase_value',  # 매입금액
                              'd2_deposit'  # D+2 예수금
                              ]

        # 체결 정보 컬럼명
        self.conc_columns = ['conc_time',  # 체결시간
                             'conc_date',  # 체결날짜
                             'stock_code',  # 종목코드
                             'stock_name',  # 종목명
                             'buy_sell',  # 매수매도
                             'conc_amount',  # 체결수량
                             'conc_price',  # 체결금액
                             'conc_value'  # 체결금약
                             ]

        self.db_manager = db_manager

        self.stocks = None

        self.stocks_selected = stocks_selected

        self.load_stock_codes()

        if stocks_selected is not None:
            self.stocks = self.stocks.loc[self.stocks_selected]

    def load_stock_codes(self):
        """
            [load stock codes]
        """
        kospi = self.get_stock_code_by_market('KOSPI')
        kosdaq = self.get_stock_code_by_market('KOSDAQ')

        stocks = pd.concat((kospi, kosdaq), axis=0)

        self.stocks = stocks

    def post_daily_conc_history(self):
        """
            체결 결과 DB 적재
             * 시스템 오류 대비해서 체결마다 적재 필요
        """
        try:
            query =\
                self._make_insert_query(self.db_name +
                                        "."+self.conc_table_name,
                                        self.conc_columns)
            if len(self.daily_conc_history) > 0:
                conc =\
                    pd.DataFrame(
                            self.daily_conc_history).values.tolist()

                for i, con in enumerate(conc):
                    conc[i] = tuple(con)
            self.db_manager.set_commit(query, is_many=True, rows=conc)
        except Exception as err:
            error_comment = "TradingManager - post_daily_conc_history"
            error_message = traceback.format_exc()
            raise Exception("{} / {} / {}".format(str(err),
                            error_comment, error_message))

    def twap_order(self, trading_list=None,
                   method='first', kind='buy',
                   start_time=(9, 0), last_time=(15, 10),
                   end_time=(15, 15), time_unit=60):
        """
            TWAP 주문
            Args:
                trading_list: pd.DataFrame. 매수 리스트
                    (index: 종목코드, key: [amounts, kind])
                        * amounts: 종목별 수량
                        * kind: buy or sell
                method: 매매방법
                    - first: 자기호가
                        * last_time까지 잔량이 남은 경우, end_time까지
                          전량 시장가에 청산
                        * 매 시점 별로 주문 잔량 있을 시 일괄 취소
                    - market: 시장가
                kind: 매수/매도
                    * buy: 매수
                    * sell: 매도
                start_time: 분할매매 시작 시간(시, 분)
                end_time: 잔량 일괄 시장가 매매 기한(시, 분)
                time_unit: 초 단위의 최소시간단위(기본값: 60초(1분))
        """
        start_t = datetime.datetime(self.today.year, self.today.month,
                                    self.today.day,
                                    start_time[0], start_time[1])
        last_t = datetime.datetime(self.today.year, self.today.month,
                                   self.today.day,
                                   last_time[0], last_time[1])
        end_t = datetime.datetime(self.today.year, self.today.month,
                                  self.today.day,
                                  end_time[0], end_time[1])

        time_steps = round((last_t - start_t).seconds / time_unit)
        tdelta = datetime.timedelta(seconds=time_unit)

        # 매매 시점 리스트 생성
        timeline = []
        for t in range(time_steps):
            step_t = start_t + (t * tdelta)
            timeline.append(step_t)

        # sell 먼저 주문하도록 종목 순서 조정
        trading_list = trading_list.sort_values('kind', ascending=False)

        # 종목 별로 시점 별 매매수량 계산
        amounts = pd.DataFrame(index=timeline)
        for code in trading_list.index:
            step_amounts = defaultdict(None)
            total_amount = trading_list['amounts'].loc[code]

            if total_amount >= time_steps:
                step_amount = int(np.floor(total_amount / time_steps))
                res = total_amount - (step_amount * time_steps)

                for step_t in timeline:
                    step_amounts[step_t] = step_amount

                    if step_t == timeline[-1]:
                        step_amounts[step_t] += res
            else:
                step = round(time_steps / total_amount)
                step_cnts = int(np.floor(time_steps / step))

                for t_idx in range(step_cnts):
                    step_t = timeline[step * t_idx]
                    step_amounts[step_t] = 1
                    total_amount -= 1

                if total_amount > 0:
                    step_amounts[timeline[-1]] = total_amount

            amounts[code] = pd.Series(step_amounts)
        amounts = amounts.where(pd.notnull(amounts), 0).astype(int)
        proc_bl = np.zeros_like(amounts)
        proc_bl = pd.DataFrame(proc_bl, index=amounts.index,
                               columns=amounts.columns)
        last_flag = np.zeros((1, proc_bl.shape[1]))
        last_flag = pd.DataFrame(last_flag, columns=proc_bl.columns)

        time_cnt = 0
        while True:
            now = datetime.datetime.now()
            now = datetime.datetime(now.year, now.month, now.day,
                                    now.hour, now.minute, now.second)
            if timeline[time_cnt] == now:
                for code in amounts.columns:
                    if proc_bl[code].loc[now] != 1:
                        if trading_list['kind'].loc[code] == 'buy':
                            kind = 'buy'
                            kind_num = "2"
                        elif trading_list['kind'].loc[code] == 'sell':
                            kind = 'sell'
                            kind_num = "1"

                        amnt = amounts[code].loc[now]
                        if amnt > 0:
                            # 호가 정보 가져오기
                            bid_ask = self.get_bid_ask(code)
                            if method == 'first':
                                if kind == 'buy':
                                    t_price =\
                                        bid_ask[code].loc['bid_1']
                                elif kind == 'sell':
                                    t_price =\
                                        bid_ask[code].loc['ask_1']
                            elif method == 'market':
                                if kind == 'buy':
                                    t_price =\
                                        bid_ask[code].loc['ask_1']
                                elif kind == 'sell':
                                    t_price =\
                                        bid_ask[code].loc['bid_1']

                            # 종목 별 이전 미체결 주문 확인
                            for o_num, o_dict in\
                                    self.order_data.items():
                                if o_dict['종목코드'] == code and\
                                        o_dict['주문종류'] == kind_num:
                                    if o_dict['주문단가'] != t_price:
                                        self.cancel_order(o_num, code)

                            # 주문
                            self.stock_order(code, t_price, amnt, kind)
                        proc_bl[code].loc[now] = 1
                time_cnt += 1
            elif now > timeline[-1] and\
                    now < end_t:  # 잔량 일괄 시장가 매매
                for code in amounts.columns:
                    if last_flag[code].iloc[0] != 1:
                        if trading_list['kind'].loc[code] == 'buy':
                            kind = 'buy'
                            kind_num = "2"
                        elif trading_list['kind'].loc[code] == 'sell':
                            kind = 'sell'
                            kind_num = "1"

                        # 호가 정보 가져오기
                        bid_ask = self.get_bid_ask(code)
                        if kind == 'buy':
                            t_price = bid_ask[code].loc['ask_1']
                        elif kind == 'sell':
                            t_price = bid_ask[code].loc['bid_1']

                        # 종목 별 미체결 주문 확인
                        amnt = 0
                        for o_num, o_dict in self.order_data.items():
                            if o_dict['종목코드'] == code and\
                                    o_dict['주문종류'] == kind_num:
                                if o_dict['주문단가'] != t_price:
                                    amnt += o_dict['주문잔량']
                                    self.cancel_order(o_num, code)

                        # 잔량 일괄 시장가 주문
                        if amnt > 0:
                            self.stock_order(code, t_price,
                                             amnt, kind)
                        last_flag[code].iloc[0] = 1
            elif now > end_t:
                break

    def stock_order(self, code, price,
                    amount, kind,
                    order_condition="0", bid_ask_code="01"):
        """
            주식 주문 메소드
            Args:
                code: 종목코드
                price: 주문단가
                amount: 매매수량
                kind: 매매 종류
                    - 'buy': 매수
                    - 'sell': 매도
                order_condition: 주문 조건 구분 코드
                    - "0": 기본
                    - "1": IOC
                    - "2": FOK
                bid_ask_code: 주문호가 구분코드
                    - 01: 보통
                    - 03: 시장가
        """
        if kind == 'buy':
            kind_num = "2"
        elif kind == 'sell':
            kind_num = "1"

        self.objOrder.SetInputValue(0, kind_num)  # 매매 종류
        self.objOrder.SetInputValue(1, self.acc)
        self.objOrder.SetInputValue(2, self.accFlag[0])
        self.objOrder.SetInputValue(3, code)
        self.objOrder.SetInputValue(4, amount)
        self.objOrder.SetInputValue(7, order_condition)
        self.objOrder.SetInputValue(8, bid_ask_code)

        if bid_ask_code == '01':        
            self.objOrder.SetInputValue(5, price)

        # 매매 주문 요청
        self.objOrder.BlockRequest()

        rqStatus = self.objOrder.GetDibStatus()
        rqRet = self.objOrder.GetDibMsg1()
        print("통신상태", rqStatus, rqRet)
        if rqStatus != 0:
            return False

        orderdata = {}

        now = time.localtime()
        sTime =\
            "%04d-%02d-%02d %02d:%02d:%02d" %\
            (now.tm_year, now.tm_mon, now.tm_mday,
             now.tm_hour, now.tm_min, now.tm_sec)
        orderdata['주문시간'] = sTime
        orderdata['주문종류'] = self.objOrder.GetHeaderValue(0)
        orderdata['종목코드'] = self.objOrder.GetHeaderValue(3)
        orderdata['주문수량'] = self.objOrder.GetHeaderValue(4)
        orderdata['주문잔량'] = orderdata['주문수량']
        orderdata['체결수량'] = 0
        orderdata['주문단가'] = self.objOrder.GetHeaderValue(5)
        orderdata['주문번호'] = self.objOrder.GetHeaderValue(8)

        self.order_data[orderdata['주문번호']] = orderdata

    def cancel_order(self, ordernum, code, block=True):
        """
            취소 주문(전량 취소)
                Args:
                    ordernum: 주문번호
                    code: 종목코드
                    block: BlockRequest 여부
                        True: BlockRequest 사용
                        False: Request를 이용하여 취소 주문
        """
        if block:
            order_func = self.objCancelOrder.BlockRequestCancel
        else:
            order_func = self.objCancelOrder.RequestCancel

        amount = self.order_data[ordernum]['주문잔량']

        if not order_func(ordernum, code, amount):
            print("취소주문 실패")
        else:
            del self.order_data[ordernum]

    def get_jango_data(self):
        " 실시간 잔고 데이터 가져오기 "
        self.requestJango()

        for key in self.jangoData.keys():
            self.jangoData[key]['현재가'] = self.curDatas[key]['cur']
            self.jangoData[key]['평가금액'] =\
                self.jangoData[key]['현재가'] *\
                self.jangoData[key]['잔고수량']

        return pd.DataFrame(self.jangoData)

    @property
    def total_asset_value(self):
        " 보유종목평가금액 + 예수금 "
        jango = self.get_jango_data()

        if len(jango) > 0:
            stock_value =\
                (jango.loc['잔고수량'] * jango.loc['현재가']).sum()
        else:
            stock_value = 0.

        total_asset_value = stock_value + self.yesu

        return total_asset_value

    def get_bid_ask(self, codes):
        " 실시간 종목 10차호가 가져오기"
        if isinstance(codes, str):
            codes = [codes]

        bid_ask_dict = {}
        for code in codes:
            bid_ask = self.objRPCur.request_bid_ask(code)
            bid_ask_dict[code] = bid_ask

        return pd.DataFrame(bid_ask_dict)

    def StopSubscribe(self):
        if self.isSB:
            for key, obj in self.objCur.items():
                obj.Unsubscribe()
            self.objCur = {}

        self.isSB = False
        self.objConclusion.Unsubscribe()

    def requestJango(self):
        self.StopSubscribe()

        # 주식 잔고 통신
        if self.obj6033.requestJango(self) == False:
            return

        # 잔고 현재가 통신
        codes = set()
        for code, value in self.jangoData.items():
            codes.add(code)

        objMarketEye = CpMarketEye()
        codelist = list(codes)
        if objMarketEye.Request(codelist, self) == False:
            exit()

        # 실시간 현재가 요청
        cnt = len(codelist)
        for i in range(cnt):
            code = codelist[i]
            self.objCur[code] = CpPBStockCur()
            self.objCur[code].Subscribe(code, self)
        self.isSB = True

        # 실시간 주문 체결 요청
        self.objConclusion.Subscribe('', self)

        # D+2 예수금 가져오기
        self.yesu = self.requestYesu()

    def requestYesu(self):
        obj = wc.Dispatch('CpTrade.CpTd0732')
        obj.SetInputValue(0, self.acc)
        obj.SetInputValue(1, self.accFlag[0])

        obj.BlockRequest()

        yesu = obj.GetHeaderValue(66)

        return yesu

    def updateJangoCont(self, pbCont):
        """
            실시간 주문 체결 처리 로직
            * 주문 체결에서 들어온 신용 구분 값
              ==> 잔고 구분값으로 치환
        """
        dictBorrow = {
            '현금': ord(' '),
            '유통융자': ord('Y'),
            '자기융자': ord('Y'),
            '주식담보대출': ord('B'),
            '채권담보대출': ord('B'),
            '매입담보대출': ord('M'),
            '플러스론': ord('P'),
            '자기대용융자': ord('I'),
            '유통대용융자': ord('I'),
            '기타': ord('Z')
            }

        # 잔고 리스트 map의 key 값
        # key = (pbCont['종목코드'],
        #  dictBorrow[pbCont['현금신용']], pbCont['대출일'])
        code = pbCont['종목코드']

        # 접수, 거부, 확인 등은 매도 가능 수량만 업데이트 한다.
        if pbCont['체결플래그'] == '접수' or\
            pbCont['체결플래그'] == '거부' or\
            pbCont['체결플래그'] == '확인':
            if code not in self.jangoData:
                return
            self.jangoData[code]['매도가능'] = pbCont['매도가능수량']
            return

        if pbCont['체결플래그'] == '체결':
            if code not in self.jangoData:  # 신규 잔고 추가
                if pbCont['체결기준잔고수량'] == 0:
                    return
                print('신규 잔고 추가', code)

                # 신규 잔고 추가
                item = {}
                item['종목코드'] = pbCont['종목코드']
                item['종목명'] = pbCont['종목명']
                item['현금신용'] = dictBorrow[pbCont['현금신용']]
                item['대출일'] = pbCont['대출일']
                item['잔고수량'] = pbCont['체결기준잔고수량']
                item['매도가능'] = pbCont['매도가능수량']
                item['장부가'] = pbCont['장부가']

                # 매입금액 = 장부가 * 잔고수량
                item['매입금액'] = item['장부가'] * item['잔고수량']

                print('신규 현재가 요청', code)
                self.objRPCur.Request(code, self)
                self.objCur[code] = CpPBStockCur()
                self.objCur[code].Subscribe(code, self)

                item['현재가'] = self.curDatas[code]['cur']
                item['대비'] = self.curDatas[code]['diff']
                item['거래량'] = self.curDatas[code]['vol']

                self.jangoData[code] = item
            else:
                # 기존 잔고 업데이트
                item = self.jangoData[code]
                item['종목코드'] = pbCont['종목코드']
                item['종목명'] = pbCont['종목명']
                item['현금신용'] = dictBorrow[pbCont['현금신용']]
                item['대출일'] = pbCont['대출일']
                item['잔고수량'] = pbCont['체결기준잔고수량']
                item['매도가능'] = pbCont['매도가능수량']
                item['장부가'] = pbCont['장부가']

                # 매입금액 = 장부가 * 잔고수량
                item['매입금액'] = item['장부가'] * item['잔고수량']

                # 잔고 수량이 0이면 잔고 제거
                if item['잔고수량'] == 0:
                    del self.jangoData[code]
                    self.objCur[code].Unsubscribe()
                    del self.objCur[code]
        return

    def updateJangoCurPBData(self, curData):
        " 실시간 현재가 처리 로직 "
        code = curData['code']
        self.curDatas[code] = curData
        self.upjangoCurData(code)

    def upjangoCurData(self, code):
        """
            잔고에 동일 종목을 찾아 업데이트 하자
            - 현재가/대비/거래량/평가금액/평가손익
        """
        curData = self.curDatas[code]
        item = self.jangoData[code]
        item['현재가'] = curData['cur']
        item['대비'] = curData['diff']
        item['거래량'] = curData['vol']

    def kill_client(self):
        " Kill Client(새로운 자동 로그인을 위해) "
        try:
            os.system('taskkill /IM ncStarter* /F /T')
            os.system('taskkill /IM CpStart* /F /T')
            os.system('taskkill /IM DibServer* /F /T')
            os.system('wmic process where ' +
                      '"name like \'%ncStarter%\'" call terminate')
            os.system('wmic process where ' +
                      '"name like \'%CpStart%\'" call terminate')
            os.system('wmic process where ' +
                      '"name like \'%DibServer%\'" call terminate')
        except Exception as err:
            error_comment = "TradingManager - kill_client"
            error_message = traceback.format_exc()
            raise Exception("{} / {} / {}".format(str(err),
                            error_comment, error_message))

    def connect(self, account_id=None, account_pwd=None,
                account_pwdcert=None):
        """
            Cybos Plus 접속
            (id/pwd = None일 경우 class 선언 시
            입력 받은 내용 활용
        """
        try:
            if account_id is None:
                account_id = self.account_id
            if account_pwd is None:
                account_pwd = self.account_pwd
            if account_pwdcert is None:
                account_pwdcert = self.account_pwdcert

            if not self.connected():
                self.disconnect()
                self.kill_client()
                self.app = application.Application()

                path = 'C:/Daishin/starter/ncStarter.exe ' +\
                    '/prj:cp /id:{account_id} /pwd:{pwd} ' +\
                    '/pwdcert:{pwdcert} /autostart'
                self.app.start(path.format(account_id=account_id,
                                      pwd=account_pwd,
                                      pwdcert=account_pwdcert))

            while not self.connected():
                time.sleep(1)
            return True
        except Exception as err:
            error_comment = "TradingManager - connect"
            error_message = traceback.format_exc()
            raise Exception("{} / {} / {}".format(str(err),
                            error_comment, error_message))

    def init_plus_check(self):
        """
            Plus 실행 기본 체크 함수
        """
        try:
            # 프로세스가 관리자 권한으로 실행 여부
            if ctypes.windll.shell32.IsUserAnAdmin():
                print('정상: 관리자권한으로 실행된 프로세스입니다.')
            else:
                print('오류: 일반권한으로 실행됨. ' +
                      '관리자 권한으로 실행해 주세요')
                return False

            # 연결 여부 체크
            if self.g_objCpStatus.IsConnect == 0:
                print("PLUS가 정상적으로 연결되지 않음.")
                return False

            # 주문 관련 초기화
            if self.g_objCpTrade.TradeInit(0) != 0:
                print("주문 초기화 실패")
                return False

            return True
        except Exception as err:
            error_comment = 'TradingManager - init_plus_check'
            error_message = traceback.format_exc()
            raise Exception("{} / {} / {}".format(str(err),
                            error_comment, error_message))

    def connected(self):
        try:
            b_connected = self.g_objCpStatus.IsConnect
            if b_connected == 0:
                return False
            return True
        except Exception as err:
            error_comment = "TradingManager - connected"
            error_message = traceback.format_exc()
            raise Exception("{} / {} / {}".format(str(err),
                            error_comment, error_message))

    def disconnect(self):
        try:
            if self.connected():
                self.g_objCpStatus.PlusDisconnect()
        except Exception as err:
            error_comment = "TradingManager - disconnect"
            error_message = traceback.format_exc()
            raise Exception("{} / {} / {}".format(str(err),
                            error_comment, error_message))

    def wait_for_request(self):
        remain_count = self.g_objCpStatus.GetLimitRemainCount(1)
        if remain_count <= 0:
            time.sleep(
                self.g_objCpStatus.LimitRequestedRemainTime / 1000)

    def _objload(self, obj):
        """
            과거 연속 데이터 불러올때 사용하는 내부 함수
        """
        obj.BlockRequest()

        # 통신 결과 확인
        rq_status = obj.GetDibStatus()
        rq_ret = obj.GetDibMsg1()
        assert(rq_status == 0)

    def get_code_by_name(self, name):
        """
            종목명을 입력 받아
            종목코드를 불러오는 함수
            Args:
                name: 종목 명
            Return:
                code: 종목코드
        """
        try:
            return self.g_objCpStockCode.NameToCode(name)
        except Exception as err:
            error_comment = "TradingManager - get_code_by_name"
            error_message = traceback.format_exc()
            raise Exception("{} / {} / {}".format(str(err),
                            error_comment, error_message))

    def get_all_stock_info(self):
        """
            등록된 전체 종목의 종목코드, 종목명 가져오기

            Return:
                stocks: dict / key: stock code, value: stock name
        """
        try:
            stocks = {}

            i = 0
            bt = True
            while bt:
                try:
                    code = self.g_objCpStockCode.GetData(0, i)
                    name = self.g_objCpStockCode.GetData(1, i)
                except Exception as err:
                    bt = False
                finally:
                    i += 1
                    stocks[code] = name

            return stocks
        except Exception as err:
            error_comment = "TradingManager - get_all_stock_info"
            error_message = traceback.format_exc()
            raise Exception("{} / {} / {}".format(str(err),
                            error_comment, error_message))

    def get_stock_code_by_market(self, market):
        """
            시장 별 종목 정보 가져오기
            Args:
                market:
                    1: KOSPI
                    2: KOSDAQ
                    3: KONEX
            Return:
                stocks: 종목 정보 DataFrame
        """
        try:
            if market == "KOSPI":
                market_idx = 1
            elif market == "KOSDAQ":
                market_idx = 2
            elif market == "KONEX":
                market_idx = 3

            code_list =\
                self.g_objCodeMgr.GetStockListByMarket(market_idx)

            stocks = {}
            for _, code in enumerate(code_list):
                name = self.g_objCodeMgr.CodeToName(code)
                secode = self.g_objCodeMgr.GetStockSectionKind(code)

                if secode == 1:
                    secode = "Stock"
                elif secode == 10:
                    secode = "ETF"
                elif secode == 17:
                    secode = "ETN"
                stocks[code] = [secode, name]

            stocks = pd.DataFrame(stocks, index=['Class', 'Name'])
            stocks = stocks.transpose()

            return stocks
        except Exception as err:
            error_comment = "TradingManager - get_stock_code_by_market"
            error_message = traceback.format_exc()
            raise Exception("{} / {} / {}".format(str(err),
                            error_comment, error_message))

    def get_stock_status(self, codes):
        """
            종목 별 상태 확인

            Return:
                0: 정상
                1: 주의
                2: 경고
                3: 위험예고
                4: 위험
                5: 경고예고
        """
        status = {}
        for code in codes:
            st = self.g_objCodeMgr.GetStockControlKind(code)
            status[code] = {'status': st}
        status = pd.DataFrame(status).transpose()

        return status

    def get_multiple_series(self, codes, dtype='종가',
                            end_date=None,
                            c_type='D', l=None, adj='1'):
        """
            복수 종목 과거 종가 데이터 가져오는 함수
            Args:
                codes: 종목 코드 리스트
                dtype: 데이터 종류(기본값: 종가)
                start_date: 시계열 시작 날짜
                end_date: 시계열 마지막 날짜
                c_type: 'D': Day, 'W': Week, 'M': Month,
                        'm': Minute, 'T': Tick
                l: 영업일 기준 시계열 길이(기본: 250일)
                adj: 수정주가 여부('1': 수정주가, '0': 무수정주가)
            Return:
                price: 복수 종목 주가 시계열 데이터프레임
        """
        data = self._get_multiple_hist_data(codes, dtype,
                                        None, end_date,
                                        c_type, l, adj)

        if l is not None:
            if data.shape[0] < l:
                while True:
                    rem = l - data.shape[0] + 1
                    sdata =\
                        self._get_multiple_hist_data(codes, dtype,
                                                    None, data.index[0],
                                                    c_type, rem, adj)
                    data = sdata.append(data.iloc[1:])
                    if data.shape[0] >= l:
                        break

        return data

    def _get_multiple_hist_data(self, codes, dtype='종가',
                               start_date=None, end_date=None,
                               c_type='D',
                               l=250, adj='1'):
        """
            복수 종목 과거 종가 데이터 가져오는 함수
            Args:
                codes: 종목 코드 리스트
                dtype: 데이터 종류(기본값: 종가)
                end_date: 시계열 마지막 날짜
                c_type: 'D': Day, 'W': Week, 'M': Month,
                        'm': Minute, 'T': Tick
                l: 영업일 기준 시계열 길이(기본: 250일)
                adj: 수정주가 여부('1': 수정주가, '0': 무수정주가)
            Return:
                price: 복수 종목 주가 시계열 데이터프레임
        """
        if start_date is not None:
            start_date = self.convert_date(start_date)
            start_date =\
                datetime.datetime.strftime(start_date, '%Y%m%d')
        if end_date is not None:
            end_date = self.convert_date(end_date)
            end_date =\
                datetime.datetime.strftime(end_date, '%Y%m%d')

        data = {}
        length = []
        l_dict = {}
        dates = {}
        for i, code in enumerate(codes):
            p_data = self.get_hist_data(code,
                                        start_date=start_date,
                                        end_date=end_date,
                                        c_type=c_type, l=l, adj=adj)
            data[code] = p_data[dtype]
            dates[code] = p_data['날짜'].values
            length.append(data[code].shape[0])
            l_dict[i] = code

        length = np.array(length)
        l_max = np.argmax(length)
        max_code = l_dict[l_max]

        index_temp = dates[max_code]

        index = []
        for i, date in enumerate(index_temp):
            date_str = str(date)
            date_t = datetime.date(int(date_str[:4]),
                                   int(date_str[4:6]),
                                   int(date_str[6:]))
            index.append(date_t)

        ptemp = pd.DataFrame(index=data[max_code].index)
        for code in data:
            ptemp[code] = data[code]

        price = pd.DataFrame(ptemp.values, columns=ptemp.columns,
                             index=index)

        return price

    def get_hist_data(self, code, start_date=None, end_date=None,
                      c_type='D', l=None, adj='1'):
        """
            과거 데이터를 가져오는 함수
            code: 종목코드
            start_date, end_date: YYYYMMDD
            c_type: 'D': Day, 'W': Week, 'M': Month,
                    'm': Minute, 'T': Tick
            l: 시계열 길이
        """
        stock_chart = wc.Dispatch("CpSysDib.StockChart")

        # SetinputValue
        stock_chart.SetInputValue(0, code)
        if l is not None:
            stock_chart.SetInputValue(1, ord('2'))
            stock_chart.SetInputValue(4, l)
            if end_date is not None:
                stock_chart.SetInputValue(2, end_date)
        else:
            stock_chart.SetInputValue(1, ord('1'))
            stock_chart.SetInputValue(2, end_date)
            stock_chart.SetInputValue(3, start_date)
        stock_chart.SetInputValue(5, (0, 1, 2, 3, 4, 5, 8, 9, 13, 19))
        stock_chart.SetInputValue(6, ord(c_type))
        stock_chart.SetInputValue(9, ord(adj))

        stock_chart.BlockRequest()

        # GetHeaderValue
        numData = stock_chart.GetHeaderValue(3)
        numField = stock_chart.GetHeaderValue(1)

        data = []
        # GetDataValue
        for i in range(numData):
            temp = []
            for j in range(numField):
                temp.append(stock_chart.GetDataValue(j, i))
            data.append(temp)

        data = pd.DataFrame(data, columns=['날짜', '시간', '시가',
                                           '고가', '저가', '종가',
                                           '거래량', '거래대금',
                                           '시가총액',
                                           '수정주가비율'])
        data = data.sort_values(by=['날짜', '시간'])

        return data

    def get_min_data(self, code, date_list, length=381):
        """
            종목 별 분 단위 데이터를
            주어진 기간(date_list)에 대해서 요청 및 수집

            Args:
                code: 종목코드
                date_list: array
                    * date 형태: 'yyyy-mm-dd'
                length: long. 일자별 데이터 길이
                    * 종목: 381
                    * 지수: 390
            Return:
                price: pd.DataFrame
        """
        for i, date in enumerate(date_list):
            if isinstance(date, datetime.date):
                date_t = date.strftime('%Y%m%d')
            else:
                date_t = date[:4] + date[5:7] + date[8:]
            data = self.get_hist_data(code, end_date=date_t,
                                    c_type='m', l=length, adj='1')
            if i == 0:
                price = data
            else:
                price = price.append(data)

        return price

    def get_daily_single_series(self, code, end_t=None,
                            length=250, adj=True):
        """
            단일 종목(지수) 과거 일간 시계열 데이터 가져오는 함수

            Args:
                code: 종목코드(지수코드)
                end_t: 종료 시점('yyyy-mm-dd' or 'yyyymmdd')
                length: 시계열 길이
                adj: 수정 주가 계산 여부(default: True)
        """
        if adj:
            adj = '1'
        else:
            adj = '0'

        if isinstance(end_t, datetime.date):
            end_t = end_t.strftime('%Y%m%d')
        elif "-" in end_t:
            end_t = end_t[:4] + end_t[5:7] + end_t[8:]

        series = self.get_hist_data(code, end_date=end_t,
                                    c_type='D', l=length, adj=adj)

        if series.shape[0] < length:
            while True:
                end_t = series['날짜'].iloc[0]
                rem = length - series.shape[0] + 1
                h_series = self.get_hist_data(code, end_date=end_t,
                                        c_type='D', l=rem, adj=adj)
                series = h_series.append(series.iloc[1:])

                if series.shape[0] >= length or h_series.shape[0] == 1:
                    break

        series = series.set_index('날짜')
        series = series[['시가', '고가', '저가', '종가', '거래량', '거래대금']]

        return series

    def convert_date(self, date):
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        elif isinstance(date, datetime.datetime):
            date = date.date()

        return date

    def cur_price(self, code):
        cur = self.g_objCur

        cur.SetInputValue(0, code)

        cur.BlockRequest()
        price = cur.GetHeaderValue(11)

        return price

    def _make_insert_query(self, table, key_list):
        """
            ffp_common 활용 데이터 truncated insert query
            만드는 함수
            Args:
                table: str. 'db_name.table_name'
                key_list: list.
            Returns:
                insert_query: str.
        """
        str_key_column =\
            "(" + ", ".join(["'" + str(key) + "'"
                             if isinstance(key, int) else str(key)
                             for key in key_list]) + ")"
        str_values = "(" + ", ".join(["%s"] * len(key_list)) + ")"

        update_cond_skeleton = "{key} = values({key})"
        str_after_update =\
            ", ".join([update_cond_skeleton.format(
                       key="'" + str(key) + "'"
                       if isinstance(key, int) else str(key))
                       for key in key_list])

        insert_query = """
                            insert into
                                {table} {str_key_column}
                            values
                                {str_values}
                            on duplicate key update
                                {update_cond_skeleton}
                       """.format(table=table,
                                  str_key_column=str_key_column,
                                  str_values=str_values,
                                  update_cond_skeleton=str_after_update)

        return insert_query


class CpEvent:
    """
        실시간 이벤트 수신 클래스
    """
    def set_params(self, client, name, caller):
        self.client = client  # CP 실시간 통신 object
        self.name = name  # 서비스가 다른 이벤트를 구분하기 위한 이름
        self.caller = caller  # callback을 위해 보관

        # 구분값: 텍스트로 변경하기 위해 딕셔너리 이용
        self.dictflag12 = {'1': '매도', '2': '매수'}
        self.dictflag14 = {'1': '체결', '2': '확인',
                           '3': '거부', '4': '접수'}
        self.dictflag15 = {'00': '현금', '01': '유통융자',
                           '02': '자기융자', '03': '유통대주',
                           '04': '자기대주', '05': '주식담보대출',
                           '07': '채권담보대출', '06': '매입담보대출',
                           '08': '플러스론', '13': '자기대용융자',
                           '15': '유통대용융자'}
        self.dictflag16 = {'1': '정상주문', '2': '정정주문',
                           '3': '취소주문'}
        self.dictflag17 = {'1': '현금', '2': '신용', '3': '선물대용',
                           '4': '공매도'}
        self.dictflag18 = {'01': '보통', '02': '임의', '03': '시장가',
                           '05': '조건부지정가'}
        self.dictflag19 = {'0': '없음', '1': 'IOC', '2': 'FOK'}

    def OnReceived(self):
        # 실시간 처리 - 현재가 주문 체결
        if self.name == 'stockcur':
            code = self.client.GetHeaderValue(0)  # 초
            name = self.client.GetHeaderValue(1)  # 초
            timess = self.client.GetHeaderValue(18)  # 초
            exFlag = self.client.GetHeaderValue(19)  # 예상체결 플래그
            cprice = self.client.GetHeaderValue(13)  # 현재가
            diff = self.client.GetHeaderValue(2)  # 대비
            cVol = self.client.GetHeaderValue(17)  # 순간체결수량
            vol = self.client.GetHeaderValue(9)  # 거래량

            item = {}
            item['code'] = code
            item['diff'] = diff
            item['cur'] = cprice
            item['vol'] = vol

            # 현재가 업데이트
            self.caller.updateJangoCurPBData(item)
        elif self.name == 'td0314':  # 주문 Request에 대한 응답 처리
            print("[CpEvent] 주문응답")
            self.caller.OrderReplay()
            return

        # 실시간 처리 - 주문체결
        elif self.name == 'conclusion':
            # 주문 체결 실시간 업데이트
            conc = {}

            # 체결 플래그
            conc_flag = self.client.GetHeaderValue(14)
            conc['체결플래그'] = self.dictflag14[conc_flag]

            conc['주문번호'] = self.client.GetHeaderValue(5)  # 주문번호
            conc['주문수량'] =\
                self.client.GetHeaderValue(3)  # 주문/체결 수량
            conc['주문가격'] =\
                self.client.GetHeaderValue(4)  # 주문/체결 가격
            conc['원주문'] = self.client.GetHeaderValue(6)
            conc['종목코드'] = self.client.GetHeaderValue(9)
            conc['종목명'] =\
                self.caller.g_objCodeMgr.CodeToName(conc['종목코드'])

            conc['매수매도'] =\
                self.dictflag12[self.client.GetHeaderValue(12)]

            flag15 = self.client.GetHeaderValue(15)  # 신용대출구분코드
            if flag15 in self.dictflag15:
                conc['신용대출'] = self.dictflag15[flag15]
            else:
                conc['신용대출'] = '기타'

            conc['정정취소'] =\
                self.dictflag16[self.client.GetHeaderValue(16)]
            conc['현금신용'] =\
                self.dictflag17[self.client.GetHeaderValue(17)]
            conc['주문조건'] =\
                self.dictflag19[self.client.GetHeaderValue(19)]

            conc['체결기준잔고수량'] = self.client.GetHeaderValue(23)
            loandate = self.client.GetHeaderValue(20)

            if loandate == 0:
                conc['대출일'] = ''
            else:
                conc['대출일'] = str(loandate)

            flag18 = self.client.GetHeaderValue(18)
            if flag18 in self.dictflag18:
                conc['주문호가구분'] = self.dictflag18[flag18]
            else:
                conc['주문호가구분'] = '기타'

            conc['장부가'] = self.client.GetHeaderValue(21)
            conc['매도가능수량'] = self.client.GetHeaderValue(22)

            # 체결에 대한 처리
            #  미체결에서 체결이 발생한 주문번호를 찾아 주문수량과
            #  체결 수량을 비교한다.
            #  전부 미체결이면 미체결을 지우고, 부분 체결인 경우
            #  체결된 수량만큼 주문 수량에서 제한다.
            if conc_flag == "1":  # 체결
                if not conc['주문번호'] in\
                        self.caller.order_data.keys():
                    print("[CpEvent] 주문번호 찾기 실패",
                          conc['주문번호'])
                    return
                else:
                    item = self.caller.order_data[conc['주문번호']]
                    remain_amount = item['주문잔량'] - conc['주문수량']
                    if remain_amount > 0:  # 일부 체결인 경우
                        # 기존 데이터 업데이트
                        item['주문잔량'] -= conc['주문수량']
                        item['체결수량'] += conc['주문수량']

                    else:  # 전량 체결인 경우
                        del self.caller.order_data[conc['주문번호']]
                        print("체결 완료 - 주문번호: {order_num}, 종목코드: {code}".format(order_num=conc['주문번호'], code=conc['종목코드']))

                        # for debug
                        print("[CpEvent] 미체결 개수: ",
                              len(self.caller.order_data))

                    now = time.localtime()
                    now_datetime = datetime.datetime(*now[:6])
                    sTime =\
                        "%04d-%02d-%02d %02d:%02d:%02d" %\
                        (now.tm_year, now.tm_mon, now.tm_mday,
                         now.tm_hour, now.tm_min, now.tm_sec)
                    conc_history = {}
                    conc_history['체결시간'] = sTime
                    conc_history['체결날짜'] = now_datetime.date()
                    conc_history['종목코드'] = conc['종목코드']
                    conc_history['종목명'] = conc['종목명']
                    conc_history['매수매도'] = conc['매수매도']
                    conc_history['체결수량'] = conc['주문수량']
                    conc_history['체결가격'] = conc['주문가격']
                    conc_history['체결금액'] =\
                        conc['주문수량'] * conc['주문가격']

                    self.caller.daily_conc_history.append(conc_history)
                    #self.caller.post_daily_conc_history()

            # 확인에 대한 처리
            #  정정확인 - 정정주문이 발생한 원주문을 찾아
            #   부분 정정인 경우 - 기존 주문은 수량을 업데이트,
            #   새로운 정정에 의한 미체결 주문번호는 신규 추가
            #   전체 정정인 경우 - 주문 리스트의
            #                      원주문/주문번호만 업데이트
            # 취소확인 - 취소주문이 발생한 원주문을
            #            찾아 주문 리스트에서 제거한다.
            elif conc_flag == "2":  # 확인
                # 원주문 번호로 찾는다.
                if not conc['원주문'] in self.caller.order_data.keys():
                    print("[CpEvent] 원주문번호로 찾기 실패",
                          conc['원주문'])
                    # IOC/FOK의 경우 취소 주문을 낸적이 없어도
                    # 자동으로 취소 확인이 들어온다.
                    if conc['주문번호'] in\
                        self.caller.order_data.keys() and\
                            conc['정정취소'] == "3":
                        del self.caller.order_data[conc['주문번호']]

                        return
                item = self.caller.order_data[conc['원주문']]
                # 정정 확인 --> 미체결 업데이트 해야 함
                if conc['정정취소'] == "2":
                    print("[CpEvent] 정정확인", item['주문수량'],
                          conc['주문수량'])
                    amount_calc = item['주문잔량'] - conc['주문수량']
                    if amount_calc > 0:  # 일부 정정인 경우
                        # 기존 데이터 업데이트
                        item['주문수량'] -= conc['주문수량']
                        item['주문잔량'] -= conc['주문수량']

                        # 새로운 미체결 추가
                        now = time.localtime()
                        sTime = "%4d-%02d-%02d %02d:%02d:%02d" %\
                            (now.tm_year, now.tm_mon, now.tm_mday,
                             now.tm_hour, now.tm_min, now.tm_sec)

                        orderdata = {}
                        orderdata['주문시간'] = sTime
                        orderdata['주문종류'] = item['주문종류']
                        orderdata['종목코드'] = conc['종목코드']
                        orderdata['주문수량'] = conc['주문수량']
                        orderdata['주문잔량'] = conc['주문수량']
                        orderdata['체결수량'] = 0
                        orderdata['주문단가'] = conc['주문가격']
                        orderdata['주문번호'] = conc['주문번호']

                        self.caller.order_data[orderdata['주문번호']] =\
                            orderdata
                    else:  # 잔량 정정인 경우 --> 업데이트
                        item['주문번호'] = conc['주문번호']
                        item['주문수량'] = conc['주문수량']
                        item['주문잔량'] = conc['주문수량']
                        item['주문단가'] = conc['주문가격']
                        self.caller.order_data[conc['주문번호']] = item
                        del self.caller.order_data[conc['원주문']]

                # 취소확인 --> 미체결 찾아 지운다.
                elif conc['정정취소'] == "3":
                    del self.caller.order_data[conc['원주문']]
                    print("[CpEvent] 미체결 개수: ",
                          len(self.caller.order_data))

            elif conc_flag == "3":  # 거부
                print("[CpEvent] 거부")

            print(conc)
            self.caller.updateJangoCont(conc)

            return True


class CpPublish:
    """
        Plus 실시간 수신 base 클래스
    """
    def __init__(self, name, serviceID):
        self.name = name
        self.obj = wc.Dispatch(serviceID)
        self.bIsSB = False

    def __del__(self):
        self.Unsubscribe()

    def Subscribe(self, var, caller):
        if self.bIsSB:
            self.Unsubscribe()

        if len(var) > 0:
            self.obj.SetInputValue(0, var)

        handler = wc.WithEvents(self.obj, CpEvent)
        handler.set_params(self.obj, self.name, caller)
        self.obj.Subscribe()
        self.bIsSB = True

    def Unsubscribe(self):
        if self.bIsSB:
            self.obj.Unsubscribe()
        self.bIsSB = False


class CpPBStockCur(CpPublish):
    """
        CpPBStockCur: 실시간 현재가 요청 클래스
    """
    def __init__(self):
        super().__init__('stockcur', 'DsCbo1.StockCur')


class CpPBConclusion(CpPublish):
    """
        CpPBConclusion: 실시간 주문 체결 수신 플래그
    """
    def __init__(self):
        super().__init__('conclusion', 'DsCbo1.CpConclusion')


class Cp6033:
    """
        Cp6033: 주식 잔고 조회
    """
    def __init__(self):
        self.g_objCpTrade = wc.Dispatch("CpTrade.CpTdUtil")
        acc = self.g_objCpTrade.AccountNumber[0]  # 계좌번호
        accFlag = self.g_objCpTrade.GoodsList(acc, 1)  # 주식상품 구분
        print(acc, accFlag[0])

        self.objRq = wc.Dispatch('CpTrade.CpTd6033')
        self.objRq.SetInputValue(0, acc)  # 계좌번호
        self.objRq.SetInputValue(1, accFlag[0])
        self.objRq.SetInputValue(2, 50)  # 요청 건수(최대 50)
        self.dictflag1 = {ord(' '): '현금',
                          ord('Y'): '융자',
                          ord('D'): '대주',
                          ord('B'): '담보',
                          ord('M'): '매입담보',
                          ord('P'): '플러스론',
                          ord('I'): '자기융자',
                          }

    def requestJango(self, caller):
        " 실제적인 6033 통신 처리 "
        while True:
            self.objRq.BlockRequest()

            # 통신 및 통신 에러 처리
            rqStatus = self.objRq.GetDibStatus()
            rqRet = self.objRq.GetDibMsg1()

            print("통신상태", rqStatus, rqRet)

            if rqStatus != 0:
                return False

            cnt = self.objRq.GetHeaderValue(7)
            for i in range(cnt):
                item = {}
                code = self.objRq.GetDataValue(12, i)  # 종목코드
                item['종목코드'] = code
                item['종목명'] = self.objRq.GetDataValue(0, i)  # 종목명
                item['잔고수량'] = self.objRq.GetDataValue(7, i)
                item['매도가능'] = self.objRq.GetDataValue(15, i)
                item['장부가'] = self.objRq.GetDataValue(17, i)
                item['매입금액'] = item['장부가'] * item['잔고수량']

                # 잔고 추가
                caller.jangoData[code] = item

            if self.objRq.Continue == False:
                break

        return True


class CpRPCurrentPrice:
    """
        현재가 - 한종목 통신
    """
    def __init__(self):
        self.objStockMst = wc.Dispatch('DsCbo1.StockMst')

    def Request(self, code, caller):
        self.objStockMst.SetInputValue(0, code)
        ret = self.objStockMst.BlockRequest()
        if self.objStockMst.GetDibStatus() != 0:
            print("통신상태", self.objStockMst.GetDibStatus(),
                  self.objStockMst.GetDibMsg1())
            return False

        item = {}
        item['code'] = code
        item['cur'] = self.objStockMst.GetHeaderValue(11)  # 종가
        item['diff'] = self.objStockMst.GetHeaderValue(12)  # 전일대비
        item['vol'] = self.objStockMst.GetHeaderValue(18)  # 거래량
        caller.curDatas[code] = item

        return True

    def request_bid_ask(self, code):
        " request realtime bid ask data "
        self.objStockMst.SetInputValue(0, code)
        ret = self.objStockMst.BlockRequest()
        if self.objStockMst.GetDibStatus() != 0:
            print("통신상태", self.objStockMst.GetDibStatus(),
                  self.objStockMst.GetDibMsg1())
            return False

        bid_ask = defaultdict(None)

        # 10차호가 - 매도
        for i in range(9, -1, -1):
            key1 = 'ask_{num}'.format(num=i+1)
            bid_ask[key1] = self.objStockMst.GetDataValue(0, i)

        # 10차호가 - 매수
        for i in range(10):
            key2 = 'bid_{num}'.format(num=i+1)
            bid_ask[key2] = self.objStockMst.GetDataValue(1, i)

        return bid_ask


class CpMarketEye:
    """
        CpMarketEye: 복수종목 현재가 통신 서비스
    """
    def __init__(self):
        # 요청 필드 배열
        # - 종목코드,시간,대비부호,대비,현재가,거래량,종목명
        self.rqField = [0, 1, 2, 3, 4, 10, 17]  # 요청 필드

        # 관심종목 객체 구하기
        self.objRq = wc.Dispatch('CpSysDib.MarketEye')

    def Request(self, codes, caller):
        # 요청 필드 세팅
        # - 종목코드,종목명,시간,대비부호,대비,현재가,거래량
        self.objRq.SetInputValue(0, self.rqField)
        self.objRq.SetInputValue(1, codes)
        self.objRq.BlockRequest()

        # 현재가 통신 및 통신 에러 처리
        rqStatus = self.objRq.GetDibStatus()
        rqRet = self.objRq.GetDibMsg1()
        print("통신상태", rqStatus, rqRet)
        if rqStatus != 0:
            return False

        cnt = self.objRq.GetHeaderValue(2)

        for i in range(cnt):
            item = {}
            item['code'] = self.objRq.GetDataValue(0, i)  # 코드
            item['diff'] = self.objRq.GetDataValue(3, i)  # 대비
            item['cur'] = self.objRq.GetDataValue(4, i)  # 현재가
            item['vol'] = self.objRq.GetDataValue(5, i)  # 거래량

            caller.curDatas[item['code']] = item

        return True


class CpPB0314:
    """
        취소 주문 요청에 대한 응답 이벤트 처리 클래스
    """
    def __init__(self, obj):
        self.name = "td0314"
        self.obj = obj

    def Subscribe(self, caller):
        handler = wc.WithEvents(self.obj, CpEvent)
        handler.set_params(self.obj, self.name, caller)


class CpRPOrder:
    """
        주식 주문 취소 클래스
    """
    def __init__(self, caller):
        self.caller = caller
        self.acc = caller.acc
        self.accFlag = caller.accFlag
        self.objCancelOrder = wc.Dispatch("CpTrade.CpTd0314")
        self.bIsRq = False
        self.RqOrderNum = 0  # 취소 주문 중인 주문 번호

    def RequestCancel(self, ordernum, code, amount):
        """
            주문 취소 통신 - Request를 이용하여 취소 주문
            caller는 취소 주문의 reply 이벤트를 전달하기 위해 필요
        """
        if self.bIsRq:
            print("RequestCancel - 통신 중이라 주문 불가")
            return False
        print("[CpRPOrder/RequestCancel] 취소주문", ordernum,
              code, amount)
        self.objCancelOrder.SetInputValue(1, ordernum)
        self.objCancelOrder.SetInputValue(2, self.acc)
        self.objCancelOrder.SetInputValue(3, self.accFlag[0])
        self.objCancelOrder.SetInputValue(4, code)
        self.objCancelOrder.SetInputValue(5, amount)

        # 취소주문 요청
        ret = 0
        while True:
            ret = self.objCancelOrder.Request()
            if ret == 0:
                break

            print("[CpRPOrder/RequestCancel 주문 요청 실패 ret: ",
                  ret)

            if ret == 4:
                remainTime =\
                    self.caller.g_objCpStatus.LimitRequestedRemainTime
                print("연속 통신 초과에 의해 재 통신처리: ",
                      remainTime / 1000, "초 대기")
                time.sleep(remainTime / 1000)
                continue
            else:  # 1: 통신 요청 실패 3: 그 외의 오류 4: 주문개수 초과
                return False

        self.bIsRq = True
        self.RqOrderNum = ordernum

        # 주문 응답(이벤트로 수신)
        self.objReply = CpPB0314(self.objCancelOrder)
        self.objReply.Subscribe(self)

        # 취소 주문 제거
        del self.caller.order_data[ordernum]

        return True

    def BlockRequestCancel(self, ordernum, code, amount):
        """
            취소 주문 - BlockRequest를 이용해서 취소 주문
        """
        print("[CpRPOrder/BlockRequestCancel] 취소주문2",
              ordernum, code, amount)
        self.objCancelOrder.SetInputValue(1, ordernum)
        self.objCancelOrder.SetInputValue(2, self.acc)
        self.objCancelOrder.SetInputValue(3, self.accFlag[0])
        self.objCancelOrder.SetInputValue(4, code)
        self.objCancelOrder.SetInputValue(5, amount)

        # 취소주문 요청
        ret = 0
        while True:
            ret = self.objCancelOrder.BlockRequest()
            if ret == 0:
                break
            print("[CpRPOrder/BlockRequestCancel] 주문 요청 실패",
                  "ret: ", ret)
            if ret == 4:
                remainTime =\
                    self.caller.g_objCpStatus.LimitRequestedRemainTime
                time.sleep(remainTime / 1000)
                continue
            else:
                return False
        print("[CpRPOrder/BlockRequestCancel] 주문결과",
              self.objCancelOrder.GetDibStatus(),
              self.objCancelOrder.GetDibMsg1())
        if self.objCancelOrder.GetDibStatus() != 0:
            return False
        return True

    def OrderReply(self):
        """
            주문 취소 Request에 대한 응답 처리
        """
        self.bIsRq = False

        if self.objCancelOrder.GetDibStatus() != 0:
            print("[CpRPOrder/OrderReply] 통신상태",
                  self.objCancelOrder.GetDibStatus(),
                  self.objCancelOrder.GetDibMsg1())
            return False

        orderPrev = self.objCancelOrder.GetHeaderValue(1)
        code = self.objCancelOrder.GetHeaderValue(4)
        orderNum = self.objCancelOrder.GetHeaderValue(6)
        amount = self.objCancelOrder.GetHeaderValue(5)

        print("[CpRPOrder/OrderReply] 주문 취소 reply, 취소한 주문: ",
              orderPrev, code, orderNum, amount)


class Cp5339:
    """
        미체결 조회 서비스
    """
    def __init__(self, caller):
        self.caller = caller
        self.objRq = wc.Dispatch("CpTrade.CpTd5339")
        self.acc = caller.acc
        self.accFlag = caller.accFlag

    def Request5339(self, count=20):
        self.objRq.SetInputValue(0, self.acc)
        self.objRq.SetInputValue(1, self.accFlag[0])
        self.objRq.SetInputValue(4, "0")  # 전체
        self.objRq.SetInputValue(5, "1")  # 정렬 기준 - 역순
        self.objRq.SetInputValue(6, "0")  # 전체
        self.objRq.SetInputValue(7, count)  # 요청 개수 - 최대 count개

        print("[Cp5339] 미체결 데이터 조회 시작")
        # 미체결 연속 조회를 위해 while 문 사용
        while True:
            ret = self.objRq.BlockRequest()
            if self.objRq.GetDibStatus() != 0:
                print("통신상태", self.objRq.GetDibStatus(),
                      self.objRq.GetDibMsg1())
                return False

            if ret == 2 or ret == 3:
                print("통신 오류", ret)
                return False

            # 통신 초과 요청 방지에 의한 오류인 경우
            # 연속 주문오류임. 이 경우 남은 시간동안 반드시 대기해야함
            while ret == 4:
                remainTime =\
                    self.caller.g_objCpStatus.LimitRequestedRemainTime
                print("연속 통신 초과에 의해 재 통신처리: ",
                      remainTime/1000, "초 대기")
                time.sleep(remainTime / 1000)
                ret = self.objRq.BlockRequest()

            # 수신 개수
            cnt = self.objRq.GetHeaderValue(5)
            print("[Cp5339] 수신 개수", cnt)
            if cnt == 0:
                break

            for i in range(cnt):
                item = {}
                item['주문번호'] = self.objRq.GetDataValue(1, i)

                if item['주문번호'] in self.caller.order_data.keys():
                    continue
                now = time.localtime()
                sTime = "%04d-%02d-%02d %02d:%02d:%02d" %\
                        (now.tm_year, now.tm_mon, now.tm_mday,
                         now.tm_hour, now.tm_min, now.tm_sec)
                item['주문시간'] = sTime
                item['주문종류'] = self.objRq.GetDataValue(13, i)
                item['종목코드'] = self.objRq.GetDataValue(3, i)
                item['주문수량'] = self.objRq.GetDataValue(6, i)
                item['주문잔량'] = self.objRq.GetDataValue(11, i)
                item['체결수량'] = self.objRq.GetDataValue(8, i)
                item['주문단가'] = self.objRq.GetDataValue(7, i)

                self.caller.order_data[item['주문번호']] = item

            # 연속 처리 체크 - 다음 데이터가 없으면 중지
            if self.objRq.Continue == False:
                print("[Cp5339] 연속 조회 여부: 다음 데이터가 없음")
                break

        return True
