"""
    Dataset Module for Trading Decision Transformers

    @author: Younghyun Kim
    Created on 2022.03.13
"""
from dataclasses import dataclass
from pyexpat import features
import numpy as np

import torch
from torch.utils.data import Dataset


class InvestingStrategyGeneratorDataset(Dataset):
    """
        Dataset Class for Investing Strategy Generator
    """
    def __init__(self, features,
                best_st_series, worst_st_series,
                best_worst_pos_series,
                best_rebal_series, worst_rebal_series,
                indices, window=250, eps=1e-6):
        """
            Initialization

            Args:
                features: asset features
                    * dtype: np.array
                    * shape: (asset_num, date_num, feature_num)
                best_st_series: best strategy series
                    * dtype: np.array
                    * shape: (date_num)
                worst_st_series: worst strategy series
                    * dtype: np.array
                    * shape: (date_num)
                best_worst_pos_series: best worst position series
                    * dtype: np.array
                    * shape: (date_num)
                best_rebal_series: best rebalancing series
                    * dtype: np.array
                    * shape: (date_num-1, strategy_num-1, 2)
                worst_rebal_series: worst rebalancing series
                    * dtype: np.array
                    * shape: (date_num-1, strategy_num-1, 2)
                indices: index list for overall data
                    * dtype: np.array
                    * shape: (date_num_picked)
                    * date_num_picked: date_num based on rebal_pos length
                window: trailing window
                    * dtype: int
                    * default: 250
        """
        assert best_st_series.shape == worst_st_series.shape
        assert best_rebal_series.shape == worst_rebal_series.shape

        self.features = features
        self.best_st_series = best_st_series
        self.worst_st_series = worst_st_series
        self.best_worst_pos_series = best_worst_pos_series
        self.best_rebal_series = best_rebal_series
        self.worst_rebal_series = worst_rebal_series

        indices = indices[indices >= (window - 1)]
        self.indices = indices

        self.asset_num = features.shape[0]
        self.feature_num = features.shape[-1]
        self.strategy_num = self.best_rebal_series.shape[1]

        self.r_rng = np.arange(self.strategy_num)

        self.window = window
        self.eps = eps

    def normalize_features(self, features):
        """
            normalize features

            Args:
                features: feature data
                    * dtype: np.array
                    * shape: (asset_num, date_num, feature_num)
        """
        num = int(self.feature_num / 4)
        normalized = np.zeros_like(features)

        normalized[:, :, :num] =\
            self.minmax_scaling(features[:, :, :num], axis=1)
        normalized[:, :, num:2*num] =\
            self.minmax_scaling(normalized[:, :, :num], axis=0)
        normalized[:, :, 2*num:3*num] =\
            self.minmax_scaling(features[:, :, 2*num:3*num], axis=1)
        normalized[:, :, 3*num:4*num] =\
            self.minmax_scaling(features[:, :, 3*num:4*num], axis=0)

        return normalized

    def minmax_scaling(self, features, axis=0):
        """
            minmax scaling

            Args:
                features: feature data
                    * dtype: np.array
                    * shape: (date_num, asset_num, feature_num)
                axis: target axis
                    * default: 0
        """
        nmax = features.max(axis=axis, keepdims=True)
        nmin = features.min(axis=axis, keepdims=True)

        normalized = (features - nmin) / (nmax - nmin + self.eps)

        return normalized

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
            get item
        """
        id = self.indices[idx]
        bw_pos = self.best_worst_pos_series[id-(self.window-1):id+1]
        rebal_pos = bw_pos + 1

        feat_bw = self.features[:, bw_pos]
        feat_rebal = self.features[:, rebal_pos]

        features_bw = self.normalize_features(feat_bw)
        features_rebal = self.normalize_features(feat_rebal)

        best_st = self.best_st_series[bw_pos[-1]]
        worst_st = self.worst_st_series[bw_pos[-1]]

        best_r_pos = np.random.choice(self.r_rng, 1).item()
        best_rebal_st =\
            self.best_rebal_series[rebal_pos[-2], best_r_pos]

        worst_r_pos = np.random.choice(self.r_rng, 1).item()
        worst_rebal_st =\
            self.worst_rebal_series[rebal_pos[-2], worst_r_pos]

        features_bw = torch.FloatTensor(features_bw.astype(float))
        features_rebal = torch.FloatTensor(features_rebal.astype(float))

        best_st = torch.LongTensor([int(best_st)])
        worst_st = torch.LongTensor([int(worst_st)])

        best_rebal_st = torch.LongTensor(best_rebal_st.astype(int))
        worst_rebal_st = torch.LongTensor(worst_rebal_st.astype(int))

        best_idx = torch.LongTensor([0]).squeeze(-1)
        worst_idx = torch.LongTensor([1]).squeeze(-1)

        initial_idx = torch.LongTensor([0]).squeeze(-1)
        rebal_idx = torch.LongTensor([1]).squeeze(-1)

        targets = dict(
            features_bw=features_bw,
            features_rebal=features_rebal,
            best_st=best_st,
            worst_st=worst_st,
            best_rebal_st=best_rebal_st,
            worst_rebal_st=worst_rebal_st,
            best_idx=best_idx,
            worst_idx=worst_idx,
            initial_idx=initial_idx,
            rebal_idx=rebal_idx)
        return targets


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