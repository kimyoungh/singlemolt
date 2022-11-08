"""
    Knights Configs

    @author: Younghyun Kim
    Created on 2022.04.10
"""
import os
import numpy as np

import torch

from ray import tune

data_path = os.path.join(os.getcwd(), "train_data/")
model_path = os.path.join(os.getcwd(), "models/")

TT_CONFIG = {
    'factor_num': 41,
    'd_model': 16,
    'asset_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.3,
    'nhead': 4,
    'nlayers': 1,
    'activation': 'gelu',
    'pp_map_nlayers': 1,
}

TT_TUNE_CONFIG = {
    'factor_num': 41,
    'd_model': 16,
    'asset_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.3,
    'nhead': 4,
    'nlayers': 1,
    'activation': 'gelu',
    'pp_map_nlayers': 1,
    'load_model': False,
    'load_model_path': None,
    'model_path': model_path,
    'model_name': 'tt',
    'checkpoint_dir': "./ray_checkpoints/",
    'factors_path':
        data_path+"index_dt_data/factors_train.npy",
    'actions_path':
        data_path+"index_dt_data/st_series_train.npy",
    'regime_path':
        data_path+"index_dt_data/regimes_train.npy",
    'epoch_size': 1000,
    'batch_size': 32,
    'lr': tune.grid_search(
        [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.1, 1.]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 32,
    'valid_prob': 0.3,
    'num_samples': 5,
    'device': 'cuda',
    'sched_term': 5,
    'lr_decay': 0.99,
}

BERTTA_CONFIG = {
    'd_model': 32,
    'dim_ff': 32,
    'series_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'nhead': 2,
    'nlayers': 2,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'task_map_nlayers': 1,
    'max_len': 1250,
    'task_num': 249,
}

BERTTA_TUNE_CONFIG = {
    'd_model': 32,
    'dim_ff': 32,
    'series_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'nhead': 2,
    'nlayers': 2,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'task_map_nlayers': 1,
    'max_len': 1250,
    'task_num': 249,
    'load_model': False,
    'load_model_path': None,
    'model_path': model_path,
    'model_name': 'bertta',
    'checkpoint_dir': "./ray_checkpoints/",
    'series_path':
        data_path+"bertta_data/series_train.npy",
    'tasks_path':
        data_path+"bertta_data/task_train.npy",
    'epoch_size': 1000,
    'batch_size': 256,
    'lr': tune.grid_search(
        [0.001, 0.0001, 0.00001, 0.000001]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 32,
    'valid_prob': 0.3,
    'num_samples': 5,
    'device': 'cuda',
    'multiple': 10.,
}


IDT_TUNE_CONFIG = {
    'factor_num': 41,
    'd_model': 32,
    'dim_ff': 32,
    'asset_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.3,
    'idt_nhead': 4,
    'idt_nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'task_map_nlayers': 1,
    'max_len': 1250,
    'task_num': 249,
    'actions': torch.arange(5),
    'values': torch.arange(-100, 10+1),
    'rewards': torch.tensor([-10, 0, +1]),
    'K': 1,
    'value_map_nlayers': 1,
    'action_map_nlayers': 1,
    'reward_map_nlayers': 1,
    'load_model': False,
    'load_model_path': None,
    'model_path': model_path,
    'model_name': 'ipa',
    'checkpoint_dir': "./ray_checkpoints/",
    'factors_path':
        data_path+"index_dt_data/factors_train.npy",
    'values_path':
        data_path+"index_dt_data/val_idx_series_train.npy",
    'actions_path':
        data_path+"index_dt_data/st_series_train.npy",
    'rewards_path':
        data_path+"index_dt_data/rew_idx_series_train.npy",
    'regime_path':
        data_path+"index_dt_data/regimes_train.npy",
    'epoch_size': 1000,
    'batch_size': 32,
    'lr': tune.grid_search(
        [0.001]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 32,
    'valid_prob': 0.3,
    'num_samples': 5,
    'device': 'cuda',
    'multiple': 10.,
    'mul': 100.,
    'sched_term': 5,
    'lr_decay': 0.99,
}

IDT_CONFIG = {
    'factor_num': 41,
    'd_model': 32,
    'dim_ff': 32,
    'asset_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.3,
    'idt_nhead': 4,
    'idt_nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'task_map_nlayers': 1,
    'max_len': 1250,
    'task_num': 249,
    'actions': torch.arange(5),
    'values': torch.arange(-100, 10+1),
    'rewards': torch.tensor([-10, 0, +1]),
    'K': 1,
    'value_map_nlayers': 1,
    'action_map_nlayers': 1,
    'reward_map_nlayers': 1,
}

REBAL_CONFIG = {
    'factor_num': 32,
    'd_model': 32,
    'stock_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'up_map_nlayers': 1,
    'down_map_nlayers': 1,
    'market_map_nlayers': 1,
    'weights_nlayers': 1,
    'rebalancer_nlayers': 1,
    'enc_path':
        model_path+"ca_bert/ca_bert_best.pt",
}

IPA_TUNE_CONFIG = {
    'factor_num': 32,
    'd_model': 32,
    'stock_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'up_map_nlayers': 1,
    'down_map_nlayers': 1,
    'market_map_nlayers': 1,
    'port_type_num': 10,
    'port_allocator_nlayers': 1,
    'enc_path':
        model_path+"ca_bert/ca_bert_best.pt",
    'load_model': False,
    'load_model_path': None,
    'model_path': model_path,
    'model_name': 'ipa',
    'checkpoint_dir': "./ray_checkpoints/",
    'factors_path':
        data_path+"ipa_data/factors_v_idx_train.npy",
    'weights_path':
        data_path+"ipa_data/weights_v_idx_train.npy",
    'epoch_size': 1000,
    'batch_size': 32,
    'lr': tune.grid_search(
        [1e-3, 1e-4, 1e-5, 1e-6]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 32,
    'valid_prob': 0.3,
    'num_samples': 5,
    'device': 'cuda',
}

IPA_CONFIG = {
    'factor_num': 32,
    'd_model': 32,
    'stock_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'up_map_nlayers': 1,
    'down_map_nlayers': 1,
    'market_map_nlayers': 1,
    'port_type_num': 10,
    'port_allocator_nlayers': 1,
    'enc_path':
        model_path+"ca_bert/ca_bert_best.pt",
}

FT_CROSS_ASSET_BERT_CONFIG = {
    'factor_num': 32,
    'd_model': 32,
    'stock_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'up_map_nlayers': 1,
    'down_map_nlayers': 1,
    'market_map_nlayers': 1,
    'up_daily_map_nlayers': 1,
    'down_daily_map_nlayers': 1,
    'pretrained_model_path':
    model_path+"ca_bert/ca_bert_best.pt",
}

FT_CROSS_ASSET_BERT_TUNE_CONFIG = {
    'factor_num': 32,
    'd_model': 32,
    'stock_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'up_map_nlayers': 1,
    'down_map_nlayers': 1,
    'market_map_nlayers': 1,
    'up_daily_map_nlayers': 1,
    'down_daily_map_nlayers': 1,
    'load_model': False,
    'load_model_path': None,
    'pretrained_model_path':
    model_path+"ca_bert/ca_bert_best.pt",
    'model_path': model_path,
    'model_name': 'ft_cross_asset_bert',
    'checkpoint_dir': "./ray_checkpoints/",
    'factors_path':
        data_path+"ca_bert_finetuning/factors_v250_train.npy",
    'up_targets_path':
        data_path+"ca_bert_finetuning/up_targets250_train.npy",
    'down_targets_path':
        data_path+"ca_bert_finetuning/down_targets250_train.npy",
    'epoch_size': 1000,
    'batch_size': 512,
    'lr': tune.grid_search(
        [1e-6, 1e-5, 1e-4, 1e-3]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 512,
    'valid_prob': 0.3,
    'num_samples': 5,
    'device': 'cuda',
}

DIPA_TUNE_CONFIG = {
    'factor_num': 32,
    'd_model': 32,
    'stock_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'up_map_nlayers': 1,
    'down_map_nlayers': 1,
    'market_map_nlayers': 1,
    'port_type_num': 10,
    'time_num': 1250,
    'port_allocator_nlayers': 1,
    'enc_path':
        model_path+"cross_asset_bert/cross_asset_bert_best.pt",
    'load_model': False,
    'load_model_path': None,
    'model_path': model_path,
    'model_name': 'dipa',
    'checkpoint_dir': "./ray_checkpoints/",
    'factors_path':
        data_path+"dipa_data/factors_v_train.npy",
    'returns_path':
        data_path+"dipa_data/returns_v_train.npy",
    'epoch_size': 1000,
    'batch_size': 32,
    'lr': tune.grid_search(
        [1e-6, 1e-5, 1e-4, 1e-3]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 32,
    'valid_prob': 0.3,
    'num_samples': 5,
    'device': 'cuda',
    'trading_period': 250,
    'fee': 0.004,
}

CROSS_ASSET_BERT_FT_TUNE_CONFIG = {
    'factor_num': 32,
    'd_model': 32,
    'stock_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'up_map_nlayers': 1,
    'down_map_nlayers': 1,
    'market_map_nlayers': 1,
    'load_model': False,
    'load_model_path': None,
    'pretrained_model_path':
    model_path+"cross_asset_bert_r_1000/cross_asset_bert_r_1000_best.pt",
    'model_path': model_path,
    'model_name': 'cross_asset_bert_ft',
    'checkpoint_dir': "./ray_checkpoints/",
    'factors_path':
        data_path+"ca_bert_finetuning/factors_v_train.npy",
    'up_targets_path':
        data_path+"ca_bert_finetuning/up_targets_train.npy",
    'down_targets_path':
        data_path+"ca_bert_finetuning/down_targets_train.npy",
    'epoch_size': 1000,
    'batch_size': 128,
    'lr': tune.grid_search(
        [1e-6, 1e-5, 1e-4, 1e-3]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 128,
    'valid_prob': 0.3,
    'num_samples': 5,
    'device': 'cuda',
}

DIPA_TUNE_CONFIG = {
    'factor_num': 32,
    'd_model': 32,
    'stock_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'up_map_nlayers': 1,
    'down_map_nlayers': 1,
    'market_map_nlayers': 1,
    'port_type_num': 10,
    'time_num': 1250,
    'port_allocator_nlayers': 1,
    'enc_path':
        model_path+"cross_asset_bert/cross_asset_bert_best.pt",
    'load_model': False,
    'load_model_path': None,
    'model_path': model_path,
    'model_name': 'dipa',
    'checkpoint_dir': "./ray_checkpoints/",
    'factors_path':
        data_path+"dipa_data/factors_v_train.npy",
    'returns_path':
        data_path+"dipa_data/returns_v_train.npy",
    'epoch_size': 1000,
    'batch_size': 32,
    'lr': tune.grid_search(
        [1e-6, 1e-5, 1e-4, 1e-3]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 32,
    'valid_prob': 0.3,
    'num_samples': 5,
    'device': 'cuda',
    'trading_period': 250,
    'fee': 0.004,
}

DIPA_CONFIG = {
    'factor_num': 32,
    'd_model': 32,
    'stock_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'up_map_nlayers': 1,
    'down_map_nlayers': 1,
    'market_map_nlayers': 1,
    'port_type_num': 10,
    'time_num': 1250,
    'port_allocator_nlayers': 1,
    'enc_path':
        model_path+"cross_asset_bert/cross_asset_bert_best.pt",
}

CROSS_ASSET_BERT_TUNE_CONFIG = {
    'factor_num': 32,
    'd_model': 32,
    'stock_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'up_map_nlayers': 1,
    'down_map_nlayers': 1,
    'market_map_nlayers': 1,
    'load_model': False,
    'load_model_path': None,
    'model_path': model_path,
    'model_name': 'cross_asset_bert',
    'checkpoint_dir': "./ray_checkpoints/",
    'factors_path':
        data_path+"factors_v_train.npy",
    'up_targets_path':
        data_path+"up_targets_train.npy",
    'down_targets_path':
        data_path+"down_targets_train.npy",
    'market_targets_path':
        data_path+"market_updowns_train.npy",
    'epoch_size': 1000,
    'batch_size': 128,
    'lr': tune.grid_search(
        [1e-6, 1e-5, 1e-4, 1e-3]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 128,
    'valid_prob': 0.3,
    'num_samples': 5,
    'device': 'cuda',
    'factor_masking_prob': 0.3,
    'masking_prob': 0.3,
    'masking': -1,
}

CROSS_ASSET_BERT_CONFIG = {
    'factor_num': 32,
    'd_model': 32,
    'stock_embeds_map_nlayers': 1,
    'slope': 0.2,
    'dropout': 0.1,
    'nhead': 4,
    'nlayers': 4,
    'activation': 'gelu',
    'recon_map_nlayers': 1,
    'up_map_nlayers': 1,
    'down_map_nlayers': 1,
    'market_map_nlayers': 1,
}

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
    'model_path': model_path,
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
    'batch_size': 64,
    'lr': tune.grid_search(
        [1e-3, 1e-4, 1e-5, 1e-6]),
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'valid_batch_size': 64,
    'valid_prob': 0.3,
    'num_samples': 5,
    'device': 'cuda',
    'eps': 1e-6,
    'window': 250,
}