"""
    Dataset Module for Trading Decision Transformers

    @author: Younghyun Kim
    Created on 2022.03.13
"""
import numpy as np

import torch
from torch.utils.data import Dataset


class StrategyGeneratorDataset(Dataset):
    """
        Dataset Class for Strategy Generator
    """
    def __init__(self, states, action_series, pos_series,
                random_prob=0.2):
        """
            Initialization

            Args:
                states: states data
                    * dtype: np.array
                    * shape: (trade_dates, asset_num, state_dim)
                action_series: action series data
                    * dtype: np.array
                    * shape
                        (trade_dates - trading_period + 1, 2, trading_period)
                pos_series: position series data
                    * dtype: np.array
                    * shape
                        (trade_dates - trading_period + 1, trading_period)
                random_prob: random probability for arbitrary replacing
                    * dtype: float
                    * default: 0.2
        """
        assert action_series.shape[0] == pos_series.shape[0]
        assert action_series.shape[-1] == pos_series.shape[-1]

        self.states = torch.FloatTensor(states.astype(float))

        self.action_series =\
            torch.LongTensor(action_series.astype(int))
        self.pos_series = torch.LongTensor(pos_series.astype(int))

        max_action_idx = action_series.max()
        self.action_idx = np.arange(max_action_idx)

        self.random_prob = random_prob
        self.random_num = int(random_prob * self.trading_period)

    def __len__(self):
        return len(self.action_series)

    @property
    def trading_period(self):
        return self.action_series.shape[-1]

    def __getitem__(self, idx):
        """
            get item

            Args:
                idx: index
            Return:
                data: dict
                    'states': state data
                        * dtype: torch.FloatTensor
                        * shape: (-1, seq_len, asset_num, state_dim)
                    'actions': action data
                        * dtype: torch.LongTensor
                        * shape: (-1, prefer_num, seq_len)
                        * prefer_num: 2
                    'actions_masked': action data with masking
                        * dtype: torch.LongTensor
                        * shape: (-1, prefer_num, seq_len)
                        * prefer_num: 2
                    'pos': timeseries position data
                        * dtype: torch.LongTensor
                        * shape: (-1, seq_len)
        """
        actions = self.action_series[idx]
        pos = self.pos_series[idx]

        states = self.states[pos]

        actions_masked = actions.clone()

        if np.random.random(1).item() <= self.random_prob:
            action_rng = np.arange(self.trading_period)
            for i in range(2):
                mask_idx = np.random.choice(
                    action_rng, self.random_num, replace=False)

                for j in mask_idx:
                    actions_masked[i, j] =\
                        np.random.choice(self.action_idx, 1).item()

        data = {}
        data['states'] = states
        data['actions'] = actions
        data['actions_masked'] = actions_masked
        data['pos'] = pos

        return data


class TradingDTDataset(Dataset):
    """
        Dataset Class for Trading DT
    """
    def __init__(self, action_series,
                value_series, pos_series):
        """
            Initialization

            Args:
                action_series: action series data
                    * dtype: np.array
                    * shape
                        1. (trade_dates - trading_period + 1, prefer_num,
                            sample_num, trading_period)
                        2. (trade_dates - trading_period + 1,
                            prefer_num*sample_num, trading_period)
                value_series: value series data
                    * dtype: np.array
                    * shape
                        1. (trade_dates - trading_period + 1, prefer_num,
                            sample_num, trading_period)
                        2. (trade_dates - trading_period + 1,
                            prefer_num*sample_num, trading_period)
                pos_series: position series data
                    * dtype: np.array
                    * shape
                        1. (trade_dates - trading_period + 1, prefer_num,
                            sample_num, trading_period)
                        2. (trade_dates - trading_period + 1,
                            prefer_num*sample_num, trading_period)
        """
        assert action_series.shape == value_series.shape
        assert value_series.shape == pos_series.shape

        action_series =\
            torch.LongTensor(action_series.astype(int))

        self.trading_period = action_series.shape[-1]
        value_shape = value_series.shape

        if len(value_shape) == 3:
            values_best = value_series.max(1, keepdims=True)
            values_best = values_best.repeat(value_shape[1], 1)
        elif len(value_shape) == 4:
            values_best =\
                value_series.reshape(
                    value_shape[0],
                    value_shape[1]*value_shape[2], -1)
            values_best = values_best.max(1, keepdims=True).repeat(
                value_shape[1]*value_shape[2], 1)
        values_best = torch.FloatTensor(values_best.astype(float))
        self.values_best = values_best.view(-1, self.trading_period)

        value_series =\
            torch.FloatTensor(value_series.astype(float))

        pos_series = torch.LongTensor(pos_series.astype(int))

        self.action_series =\
            action_series.view(-1, self.trading_period)
        self.value_series =\
            value_series.view(-1, self.trading_period, 1)
        self.pos_series =\
            pos_series.view(-1, self.trading_period)

    def __len__(self):
        return len(self.action_series)

    @property
    def get_trading_period(self):
        return self.trading_period

    def __getitem__(self, idx):
        """
            get item

            Args:
                idx: index
            Return:
                actions: action data
                    * dtype: torch.LongTensor
                    * shape: (-1, seq_len)
                values: value data
                    * dtype: torch.FloatTensor
                    * shape: (-1, seq_len, 1)
                pos: timeseries position data
                    * dtype: torch.LongTensor
                    * shape: (-1, seq_len)
                values_best: value best data
                    * dtype: torch.FloatTensor
                    * shape: (-1, seq_len)
        """
        actions = self.action_series[idx]
        values = self.value_series[idx]
        pos = self.pos_series[idx]

        values_best = self.values_best[idx]

        return actions, values, pos, values_best