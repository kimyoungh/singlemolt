"""
    Configs for Strategy Generator Models

    @author: Younghyun Kim
    Created on 2022.03.18
"""
SIMPLE_STRATEGY_GENERATOR_CONFIG = {
    'state_dim': 33,
    'map_nlayers': 3,
    'slope': 0.2,
    'dropout': 0.2,
    'd_model': 32,
    'enc_nheads': 4,
    'enc_nlayers': 4,
    'enc_dim_ff': 32,
    'enc_activation': 'gelu',
    'dec_nheads': 4,
    'dec_nlayers': 4,
    'dec_dim_ff': 32,
    'dec_activation': 'gelu',
    'out_map_nlayers': 1,
    'strategy_num': 6,
    'max_len': 4096,
}

SSG_TRAINER_CONFIG = {
    'epochs': 20000,
    'batch_size': 64,
    'lr': 0.000001,
    'scheduling': False,
    'lr_decay': 0.95,
    'decay_steps': 5000,
    'amsgrad': True,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'model_path': './models/ssg/',
    'model_name': 'ssg',
    'logdir': './logdirs/ssg/',
    'device': 'cuda:0',
    'load_model': False,
    'load_model_path': None,
    'clip_grad': 0.5,
    'eps': 1e-6,
    'random_prob': 0.2,
    'valid_prob': 0.2,
    'valid_batch_size': 32,
}