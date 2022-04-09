"""
    상장 종목 종가 및 거래대금 시계열 가져와서 저장하는 스크립트

    @Date: 2020.02.28
    @Author: Younghyun Kim
"""
import argparse
import datetime
import numpy as np
import pandas as pd

from merlin.cfg.configs_secret import account_info
from merlin.cfg.configs import market_codes, trading_config
from merlin.trading_manager import TradingManager

class GetHistData:
    """
        상장 종목 데이터 가져오는 클래스
    """

    def __init__(self, acc_info, load_codes=True,
                stocks_req=['A069500', 'A114800', 'A122630',
                            'A229200', 'A233740', 'A251340'],
                window=20, num_pick=200, targets='stocks'):
        " initialization "

        self.today = datetime.datetime.now()
        self.prev = self.today - datetime.timedelta(days=1)

        self.today = self.today.strftime('%Y-%m-%d')
        self.prev = self.prev.strftime('%Y-%m-%d')

        self.stocks_req = stocks_req
        self.window = window
        self.num_pick = num_pick

        if targets == 'stocks':
            market = False
        elif targets == 'market':
            market = True

        self.trading_manager = TradingManager(acc_info,
                                            auto_login=True)

        if load_codes:
            self.codes = self.get_codes(clength=window, num_pick=num_pick)
        else:
            self.codes = None

    def get_codes(self, clength=250, num_pick=200):
        """
            get codes

            Args:
                clength: 거래대금 평균 계산을 위한 기간
                num_pick: 최종적으로 선택하는 종목 개수
        """
        kospi = self.trading_manager.get_stock_code_by_market('KOSPI')
        kosdaq = self.trading_manager.get_stock_code_by_market('KOSDAQ')

        codes_raw =\
            kospi[kospi['Class'].isin(['Stock', 'ETF'])].index.values.tolist()
        codes_raw += kosdaq[kosdaq['Class'] == 'Stock'].index.values.tolist()

        status = self.trading_manager.get_stock_status(codes_raw)
        codes_raw = status[status['status'] == 0].index.tolist()

        turnover = self.get_multiple_hist_data(codes_raw, dtype='거래대금',
                                               l=clength)
        tmean = turnover.mean()
        tmean = tmean.sort_values(ascending=False)

        codes = tmean.index[:num_pick].values.tolist()
        codes += self.stocks_req
        codes = np.unique(codes)

        return codes

    def get_multiple_hist_data(self, codes, dtype='종가',
                               end_date=None, l=250, adj='1'):
        " 과거 데이터 Cybos Plus API를 통해 불러오는 함수 "
        data =\
            self.trading_manager.get_multiple_series(codes, dtype=dtype,
                                                     end_date=end_date,
                                                     l=l, adj=adj)

        return data


if __name__ == "__main__":
    parser =\
        argparse.ArgumentParser(description="Cybos API를 활용하여 과거 데이터를 받는 스크립트")

    parser.add_argument('--targets', type=str, default='stocks', help='stocks or markets')
    parser.add_argument("--length", type=int, default=1000, help="series length")
    parser.add_argument("--window", type=int, default=20,
                        help="window for averaging")
    parser.add_argument("--num_pick", type=int, default=200,
                        help="number of stock picked")
    parser.add_argument('--stock_selected', action='store_true')

    args = parser.parse_args()

    if args.targets == 'stocks':
        load_codes = True
    else:
        load_codes = False

    ghd = GetHistData(account_info, load_codes=load_codes,
                      window=args.window, num_pick=args.num_pick,
                      stocks_req=['A069500', 'A114800', 'A122630',
                                  'A229200', 'A233740', 'A251340'])

    if args.targets == 'stocks':
        if args.stock_selected:
            codes = trading_config['stocks_selected']
        else:
            codes = ghd.codes

        price = ghd.get_multiple_hist_data(codes, dtype='종가',
                                        end_date=ghd.today, l=args.length)
        turnover = ghd.get_multiple_hist_data(codes, dtype='거래대금',
                                            end_date=ghd.today, l=args.length)
        mktcap = ghd.get_multiple_hist_data(codes, dtype='시가총액',
                                            end_date=ghd.today, l=args.length)

        price.to_csv("./data/price_"+str(ghd.today)+".csv")
        turnover.to_csv("./data/turnover_"+str(ghd.today)+".csv")
        mktcap.to_csv("./data/mktcap_"+str(ghd.today)+".csv")

    elif args.targets == 'markets':
        markets = pd.Series(market_codes).values.ravel()

        indices = ghd.get_multiple_hist_data(markets, dtype='종가',
                                             end_date=ghd.today, l=args.length)
        
        indices.to_csv("./data/indices_"+str(ghd.today)+".csv")