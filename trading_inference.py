"""
    포트폴리오 비중 산출 스크립트

    Created on 2021.10.28
    @author: Younghyun Kim
"""
import os
import time
import datetime
import numpy as np
import pandas as pd

import torch

from data_processing.data_processor import DataProcessor
from trading_models.yh_trading_policy_maker import YHTradingPolicyMaker
from trading_models.yh_trading_policy_maker_configs\
    import YH_TP_CONFIG
from stock_auto_trading.configs import trading_config


class TradingInference:
    """
        포트폴리오 비중 산출 및 매매 내역 산출 스크립트
    """
    def __init__(self, config=None, trader_config=None):
        " Initialization "
        if config is None:
            config = trading_config

        if trader_config is None:
            trader_config = YH_TP_CONFIG

        today = datetime.datetime.now()
        self.today = datetime.datetime(today.year, today.month,
                                        today.day)

        self.config = config
        self.trader_config = trader_config

        self.stock_info_path = self.config['stock_info_path']
        self.price_path = self.config['price_data_path']
        self.turnover_path = self.config['turnover_data_path']
        self.mktcap_path = self.config['mktcap_data_path']
        self.jango_path = self.config['jango_data_path']
        self.tav_path = self.config['total_asset_value_path']

        self.orders_path = self.config['trading_orders_path']

        self.device = self.config['device']

        self.model_path = self.config['model_path']

        self.load_model()

        quit_time = self.config['quit_time']
        quit_time = quit_time.split(':')
        self.quit_time =\
            datetime.datetime(today.year, today.month, today.day,
                            int(quit_time[0]), int(quit_time[1]))

        self.rebal_prob = self.config['rebal_prob']
        self.buffer = self.config['buffer']
        self.topk = self.config['topk']
        self.sec = self.config['sec_delay']

        self.dp = DataProcessor()

        self.rec_price = None

    def run_ordering_decision(self):
        """
            [정해진 시간에 매매에 필요한 주문서 생성하여 반환]
        """
        while True:
            if os.path.exists(self.tav_path):
                time.sleep(self.sec)
                orders = self.get_orders()
                orders.to_csv(self.orders_path)

                if os.path.exists(self.tav_path+'.old'):
                    os.remove(self.tav_path+'.old')
                os.rename(self.tav_path, self.tav_path+'.old')

                if os.path.exists(self.price_path+'.old'):
                    os.remove(self.price_path+'.old')
                os.rename(self.price_path, self.price_path+'.old')

                if os.path.exists(self.turnover_path+'.old'):
                    os.remove(self.turnover_path+'.old')
                os.rename(self.turnover_path, self.turnover_path+'.old')

                if os.path.exists(self.mktcap_path+'.old'):
                    os.remove(self.mktcap_path+'.old')
                os.rename(self.mktcap_path, self.mktcap_path+'.old')

                if os.path.exists(self.jango_path+'.old'):
                    os.remove(self.jango_path+'.old')
                os.rename(self.jango_path, self.jango_path+'.old')

            now = datetime.datetime.now()
            if now >= self.quit_time:
                break

    def get_orders(self):
        """
            [매수 매도 주문서 작성]

            Return:
                orders: {pd.DataFrame}
                    * +: 매수
                    * -: 매도
        """
        jango = self.get_jango()
        tav = jango['평가금액'].sum() * self.rebal_prob

        scores = self.get_stock_scores()
        stock_info = pd.read_csv(self.stock_info_path, index_col=0, header=0)

        nocash_index = jango.index[jango.index != 'cash']

        stocks = nocash_index.append(scores.index).unique()

        orders = pd.concat((stock_info.loc[stocks, 'Name'],
                            self.rec_price.loc[stocks],
                            scores['scores'],
                            jango['매도가능'].loc[nocash_index].\
                                rename('amounts')), axis=1)
    
        orders = orders.loc[stocks]

        new_amounts = np.floor(orders['scores'] * tav / orders['close'])
        new_amounts = new_amounts.fillna(0)

        out_port_weights = orders['amounts'] * orders['close']
        out_port_weights /= jango['평가금액'].sum()

        out_amounts = np.ceil(out_port_weights * tav / orders['close'])
        out_amounts = out_amounts.fillna(0)

        out_amounts =\
            out_amounts.where(out_amounts <= orders['amounts'], orders['amounts'])
        out_amounts = out_amounts.fillna(0)

        net_amounts = new_amounts - out_amounts

        orders = pd.concat((orders, out_amounts.rename('out_amounts'),
                            new_amounts.rename('new_amounts'),
                            net_amounts.rename('trading_amounts')),
                        axis=1)

        orders = orders[orders['trading_amounts'] != 0]
        orders = orders.sort_values('trading_amounts')
        orders = orders.fillna(0)

        return orders

    def get_stock_scores(self):
        """
            종목 스코어 산출
        """
        price, _, _ = self.get_raw_data()

        factors_t, _ =\
            self.calc_input_factors(price)

        factors_t = torch.FloatTensor(factors_t.astype(float))
        factors_t = factors_t.contiguous()
        factors_t = factors_t.to(self.device)

        jango = self.get_jango()

        wrec = self.calc_recent_weights(price.columns, jango)
        wrec =\
            wrec.values.astype(float).reshape(factors_t.shape[0], -1)
        wrec = torch.FloatTensor(wrec).to(self.device)

        with torch.no_grad():
            scores = self.trader(factors_t)
        scores = scores.cpu().numpy().reshape(-1, 1)
        scores = pd.DataFrame(scores, index=price.columns,
                            columns=['scores'])

        results = scores.sort_values('scores', ascending=False)
        results = results.iloc[self.topk]
        results['scores'] /= results['scores'].sum()

        return results

    def get_raw_data(self):
        """
            [get raw data for calculation of input factors]
        """
        price = pd.read_csv(self.price_path, index_col=0, header=0)
        turnover = pd.read_csv(self.turnover_path, index_col=0, header=0)
        mktcap = pd.read_csv(self.mktcap_path, index_col=0, header=0)

        price = price.dropna(1)
        turnover = turnover[price.columns]
        mktcap = mktcap[price.columns]

        self.rec_price = price.iloc[-1].rename('close')

        return price, turnover, mktcap

    def calc_input_factors(self, price):
        """
            당일 포트 추출을 위한 모델 입력값 계산
        """
        factors_t, trade_dates_t =\
            self.dp.calc_factor_series(price,
                                    calc_targets=False,
                                    target_date=price.index[-1])

        return factors_t, trade_dates_t

    def get_tav(self):
        """
            전체 투자금액 가져오기
        """
        with open(self.tav_path, 'r') as f:
            tav = f.read()

        return float(tav)

    def load_model(self):
        " load model "
        self.trader = YHTradingPolicyMaker(self.trader_config)
        self.trader.load_state_dict(torch.load(self.model_path,
                                            map_location=self.device))
        self.trader.eval()

    def get_jango(self):
        " get jango "
        jango = pd.read_csv(self.jango_path, index_col=0, header=0)

        jango = jango.transpose()
        jango = jango[['종목명', '매도가능', '현재가', '평가금액']]
        jango['매도가능'] = jango['매도가능'].astype(int)
        jango['현재가'] = jango['현재가'].astype(int)
        jango['평가금액'] = jango['평가금액'].astype(float)

        tav = self.get_tav() * (1. - self.buffer)
        cash = tav - jango['평가금액'].sum()
        jango.loc['cash'] = None
        jango.loc['cash', '평가금액'] = cash

        return jango

    def calc_recent_weights(self, stock_cols, jango):
        """
            최근 보유 종목 비중 계산
            get_jango를 통해 계산된 jango를 입력받아야 함

            Args:
                stock_cols: 종목코드 목록
                jango: 잔고
            * 무조건 stock_cols에는 잔고 종목들이 포함되어야 함
        """
        nocash = jango.index.isin(['cash'])
        jango = jango[~nocash]

        weights =\
            jango['평가금액'] / jango['평가금액'].sum()
        jango = pd.concat((jango, weights.rename('weights')), axis=1)

        wrec = jango.loc[jango.index, 'weights']
        wrec = wrec.reindex(index=stock_cols)
        wrec = wrec.fillna(0)

        return wrec


if __name__ == "__main__":
    ti = TradingInference(config=trading_config,
                        trader_config=YH_TP_CONFIG)
    ti.run_ordering_decision()
