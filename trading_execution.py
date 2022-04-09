"""
    Trading Execution(Python 32-bit)

    Created on 2021.10.30
    @author: Younghyun Kim
"""
import os
import pickle
import time
import traceback
import datetime
import numpy as np
import pandas as pd

from merlin.trading_manager import TradingManager
from merlin.cfg.configs import trading_config


class TradingExecution:
    """
        실시간 주식 정보 산출 및 매매 스크립트
        매일 장전 실행 및 상태 유지(08:00 정도)
    """
    def __init__(self, config=None):
        " Initialization "
        if config is None:
            config = trading_config

        self.config = config
        self.price_path = config['price_data_path']
        self.turnover_path = config['turnover_data_path']
        self.mktcap_path = config['mktcap_data_path']

        self.jango_path = config['jango_data_path']
        self.tav_path = config['total_asset_value_path']
        self.stock_info_path = config['stock_info_path']

        self.trading_orders_path = config['trading_orders_path']

        today = datetime.datetime.now()
        self.today = datetime.datetime(today.year, today.month, today.day)
        self.today_str = self.today.strftime('%Y-%m-%d')

        quit_time = config['quit_time']
        quit_time = quit_time.split(':')
        self.quit_time =\
            datetime.datetime(today.year, today.month, today.day,
                            int(quit_time[0]), int(quit_time[1]))

        self.ba_code = config['bid_ask_code']

        self.stock_num_pick = config['stock_num_pick']
        self.decision_window = config['decision_window']
        self.inference_time = config['inference_time']

        infer_time = self.inference_time.split(':')
        self.infer_time =\
            datetime.datetime(today.year, today.month, today.day,
                            int(infer_time[0]), int(infer_time[1]))

        self.input_data_length = config['input_data_length']

        self.stocks_selected = config['stocks_selected']

        self.manager = TradingManager(load_stocks=True,
                            stocks_selected=self.stocks_selected)
        self.jango = self.manager.get_jango_data()

        stocks = self.manager.stocks
        stocks.to_csv(self.stock_info_path)

        stocks = stocks[stocks['Class'].isin(['Stock', 'ETF', 5])].index.values
        self.init_stocks = stocks

        self.stocks_sel = self.get_trading_codes()

        self.get_initial_inputs()

        self.tav = self.manager.total_asset_value

        self.c_period = config['checking_period']
        self.sec = config['sec_delay']
        self.osec = config['order_delay']
        self.order_delay = config['order_delay_count']

    def run_operation(self):
        """
            실시간 input data 처리 및 매매 처리
        """
        try:
            infer = True
            check = True
            while True:
                now = datetime.datetime.now()
                if now >= self.infer_time:
                    if infer:
                        self.proc_infer_inputs()

                        self.jango = self.manager.get_jango_data()
                        self.jango.to_csv(self.jango_path)

                        self.tav = self.manager.total_asset_value
                        with open(self.tav_path, 'w') as f:
                            f.write(str(self.tav))
                        infer = False

                    if os.path.exists(self.trading_orders_path):
                        time.sleep(self.sec)
                        orders =\
                            pd.read_csv(self.trading_orders_path,
                                        header=0, index_col=0)
                        self.execute_order(orders)
                        if os.path.exists(self.trading_orders_path+'.old'):
                            os.remove(self.trading_orders_path+'.old')
                        os.rename(self.trading_orders_path,
                                self.trading_orders_path+'.old')

                if now.minute % self.c_period == 0:
                    if check:
                        time.sleep(self.sec)
                        self.tav = self.manager.total_asset_value
                        check = False
                else:
                    check = True

                if now >= self.quit_time:
                    break
        except Exception as err:
            error = traceback.format_exc()
            with open('./error.log', 'ab') as f:
                pickle.dump(error, f)

    def execute_order(self, orders):
        """
            주문 집행
        """
        cnt = 0
        for i in range(orders.shape[0]):
            code = orders.index[i]
            tamount = orders['trading_amounts'].iloc[i]
            close = orders['close'].iloc[i]
            if tamount < 0:
                amount = -tamount
                kind = 'sell'
            else:
                amount = tamount
                kind = 'buy'

            self.manager.stock_order(code, close,
                                    amount, kind,
                                    bid_ask_code=self.ba_code)
            cnt += 1

            if cnt == self.order_delay:
                time.sleep(self.osec)
                cnt = 0

    def get_trading_codes(self):
        """
            오늘 매매 대상 종목 목록 정리
            거래대금 상위 stock_num_pick개 종목 선정
        """
        status = self.manager.get_stock_status(self.init_stocks.tolist())
        codes_f = status[status['status'] == 0].index.tolist()

        turnover = self.manager.get_multiple_series(codes_f, dtype='거래대금',
                                                    l=self.decision_window+1)
        turnover = turnover.iloc[:-1]
        tmean = turnover.mean()
        tmean = tmean.sort_values(ascending=False)

        stocks = tmean.index[:self.stock_num_pick].values.tolist()
        stocks = np.unique(np.append(stocks, self.jango.columns.values))

        return stocks

    def get_initial_inputs(self):
        """
            Model Inference에 필요한 사전 데이터 불러오기
        """
        self.price =\
            self.manager.get_multiple_series(self.stocks_sel,
                                            dtype='종가',
                                            end_date=self.today_str,
                                            c_type='D',
                                            l=self.input_data_length+1)

        self.turnover =\
            self.manager.get_multiple_series(self.stocks_sel,
                                            dtype='거래대금',
                                            end_date=self.today_str,
                                            c_type='D',
                                            l=self.input_data_length+1)

        self.mktcap =\
            self.manager.get_multiple_series(self.stocks_sel,
                                            dtype='시가총액',
                                            end_date=self.today_str,
                                            c_type='D',
                                            l=self.input_data_length+1)

    def proc_infer_inputs(self):
        """
            정해진 시간에 모델에 사용할 최신 input data 가공 및 저장
        """
        price_now =\
            self.manager.get_multiple_series(self.stocks_sel,
                                            dtype='종가',
                                            end_date=self.today_str,
                                            c_type='D',
                                            l=1)

        turnover_now =\
            self.manager.get_multiple_series(self.stocks_sel,
                                            dtype='거래대금',
                                            end_date=self.today_str,
                                            c_type='D',
                                            l=1)

        mktcap_now =\
            self.manager.get_multiple_series(self.stocks_sel,
                                            dtype='시가총액',
                                            end_date=self.today_str,
                                            c_type='D',
                                            l=1)

        price = self.price.append(price_now)
        price = price[~price.index.duplicated(keep='last')]
        self.price = price.iloc[-self.input_data_length:]

        turnover = self.turnover.append(turnover_now)
        turnover = turnover[~turnover.index.duplicated(keep='last')]
        self.turnover = turnover.iloc[-self.input_data_length:]

        mktcap = self.mktcap.append(mktcap_now)
        mktcap = mktcap[~mktcap.index.duplicated(keep='last')]
        self.mktcap = mktcap.iloc[-self.input_data_length:]

        self.price.to_csv(self.price_path)
        self.turnover.to_csv(self.turnover_path)
        self.mktcap.to_csv(self.mktcap_path)


if __name__ == "__main__":
    te = TradingExecution(config=trading_config)
    te.run_operation()
