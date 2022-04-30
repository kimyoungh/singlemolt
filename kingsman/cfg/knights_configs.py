"""
    Knights Configs

    @author: Younghyun Kim
    Created on 2022.04.10
"""
import numpy as np
from ray import tune

TRADING_BERT_CONFIG = {
    'factor_num': 20,
    'asset_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'max_len': 4096,
    'd_model': 16,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
}

GLEN_SCOTIA_CONFIG = {
    'factor_num': 20,
    'asset_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'max_len': 4096,
    'd_model': 16,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'strategy_num': 7,
    'gen_num_layers': 1,
}

KOR_STRATEGY_WEIGHTS = {
    'buy_and_hold': None,
    'k200': np.array([1., 0., 0., 0.]),
    'kq': np.array([0., 1., 0., 0.]),
    'k200_i': np.array([0., 0., 1., 0.]),
    'kq_i': np.array([0., 0., 0., 1.]),
    'k200-kq': np.array([0.5, 0., 0., 0.5]),
    'kq-k200': np.array([0., 0.5, 0.5, 0.]),
}

data_path = "./train_data/"

GLEN_SCOTIA_TUNE_CONFIG = {
    'factor_num': 20,
    'asset_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'max_len': 4096,
    'd_model': 16,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'strategy_num': 7,
    'gen_num_layers': 1,
    'load_model': False,
    'load_model_path': None,
    'model_path': "./models/",
    'model_name': "glen_scotia",
    'checkpoint_dir': "./ray_checkpoints/",
    'best_worst_pos_series_path':
        data_path+"best_worst_pos_series_train.npy",
    'features_path':
        data_path+"features_train.npy",
    'best_st_series_path':
        data_path+"best_st_series_train.npy",
    'worst_st_series_path':
        data_path+"worst_st_series_train.npy",
    'best_rebal_series_path':
        data_path+"best_rebal_series_train.npy",
    'worst_rebal_series_path':
        data_path+"worst_rebal_series_train.npy",
    'epoch_size': 1000,
    'batch_size': 32,
    'lr': tune.grid_search([1e-3, 1e-4, 1e-5, 1e-6]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 32,
    'valid_prob': 0.3,
    'num_samples': 5,
    'device': 'cuda',
    'eps': 1e-6,
    'window': 250,
}