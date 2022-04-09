"""
    Configs
"""
import numpy as np

market_codes = {
    'KOSPI': 'U001',
    'KOSPI200': 'U180',
    'KOSDAQ': 'U201',
    'KOSDAQ150': 'U390',
    'MSCI_KOR': 'U530',
}

stocks_selected = ['A005930', 'A000660', 'A035420', 'A051910',
                    'A005380', 'A011200', 'A066570', 'A068270',
                    'A006400', 'A034020', 'A005490', 'A009150',
                    'A003490', 'A017670', 'A034220', 'A015760',
                    'A006280', 'A051900', 'A004020', 'A090430',
                    'A008770', 'A009540', 'A010950', 'A000720',
                    'A033780', 'A069500', 'A229200', 'A114800',
                    'A251340', 'A088980']

trading_config = {
    'price_data_path': './trading_data/price_rt.csv',
    'turnover_data_path': './trading_data/turnover_rt.csv',
    'mktcap_data_path': './trading_data/mktcap_rt.csv',
    'jango_data_path': './trading_data/jango_rt.csv',
    'total_asset_value_path': './trading_data/tav_rt.txt',
    'stock_info_path': './trading_data/stock_info.csv',
    'trading_orders_path': './trading_data/trading_orders.csv',
    'device': 'cpu',
    'model_path':
    './trading_models/models/yh_tp_trader/yh_tp_trader81.pt',
    'quit_time': '18:00',
    'rebal_prob': 1.,
    'buffer': 0.01,
    'bid_ask_code': "03",
    'stock_num_pick': 200,
    'decision_window': 250,
    'input_data_length': 501,
    'inference_time': '15:10',
    'checking_period': 30,
    'logger_path': './trading.log',
    'sec_delay': 5,
    'order_delay': 16,
    'order_delay_count': 20,
    'push': True,
    'etf_list': ['A069500', 'A114800', 'A251340', 'A229200',
                'A305720', 'A091180', 'A261220', 'A091170',
                'A091160', 'A228800', 'A228810', 'A326240',
                'A157490', 'A117680', 'A143860', 'A139220'],
    'stocks_selected': stocks_selected,
    'topk': np.arange(len(stocks_selected)),
}
