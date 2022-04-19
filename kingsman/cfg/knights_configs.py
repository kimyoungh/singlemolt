"""
    Knights Configs

    @author: Younghyun Kim
    Created on 2022.04.10
"""
TRADING_BERT_CONFIG = {
    'factor_num': 10,
    'asset_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'max_len': 4096,
    'd_model': 16,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
}