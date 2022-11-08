"""
    Dataset Module for Trading Decision Transformers

    @author: Younghyun Kim
    Created on 2022.03.13
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from re import A
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class TTDataset(Dataset):
    """
        Dataset Class for TT
    """
    def __init__(self, factors, actions, regimes, indices):
        """
            Initialization

            Args:
                factors: multifactor scores
                    * dtype: np.array(float)
                    * shape: (full_date_num, asset_num, factor_num)
                actions: action index series
                    * dtype: np.array(int)
                    * shape: (indices_num, 1, period_num)
                regimes: regime series
                    * dtype: np.array(int)
                    * shape: (indices_num, regime_num)
                        * 0: regime up
                        * 1: regime neutral
                        * 2: regime down
                indices: index array
                    * dtype: np.array(int)
                    * shape: (indices_num)
        """
        self.indices_num, _, self.period_num = actions.shape
        _, self.asset_num, self.factor_num = factors.shape

        self.factors = factors
        self.actions = actions.reshape(-1, self.period_num)
        self.regimes = regimes
        self.indices = indices

        self.regime_num = regimes.shape[1]
        self.regime_rng = np.arange(self.regime_num)

        self.regime_idx = defaultdict(np.array)

        for r in self.regime_rng:
            self.regime_idx[r] = np.argwhere(
                regimes[:, r] == 1).ravel()

        self.period_rng = np.arange(self.period_num).reshape(-1, 1)

        assert indices.shape[0] == self.indices_num
        assert self.period_num == 2

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
            get item

            Return:
                factors: multifactor scores
                    * dtype: torch.FloatTensor
                    * shape: (-1, regime_num*2, asset_num, factor_num)
                actions: action series
                    * dtype: torch.LongTensor
                    * shape: (-1, regime_num*2)
                trading_positions: trading positions
                    * dtype: torch.LongTensor
                    * shape: (-1, regime_num*2, asset_num)
                    * values
                        * 0: Zero Position
                        * 1: Long Position
        """
        reg = self.regimes[idx].argmax()
        no_regs = list(set(self.regime_rng).difference([reg]))

        indices = [idx]
        # pick regimes
        for r in no_regs:
            picked = np.random.choice(self.regime_idx[r], 1).item()
            indices.append(picked)

        fidx = self.indices[indices].reshape(1, -1) + self.period_rng

        factors = torch.FloatTensor(
            self.factors[fidx].astype(float)).transpose(0, 1).contiguous()
        factors = factors.view(-1, self.asset_num, self.factor_num)

        actions = torch.LongTensor(self.actions[indices].astype(int))

        tp_next = F.one_hot(actions[:, 0],
            num_classes=self.asset_num).view(-1, 1, self.asset_num)
        tp_init = torch.zeros_like(tp_next)

        tp = torch.cat((tp_init, tp_next), dim=1)
        trading_positions = tp.view(-1, self.asset_num)

        actions = actions.view(-1)

        return factors, actions, trading_positions


class BERTTADataset(Dataset):
    """
        Dataset Class for BERTTA
    """
    def __init__(self, series, tasks):
        """
            Initialization

            Args:
                series: time series of multiple assets
                    * dtype: np.array
                    * shape: (series_num, seq_len)
                tasks: task answers for multiple asset time series
                    * dtype: np.array
                    * shape: (series_num, task_num)
        """
        if len(series.shape) > 2:
            series = series.reshape(-1, series.shape[-1])

        if len(tasks.shape) > 2:
            tasks = tasks.reshape(-1, tasks.shape[-1])

        self.series = series
        self.tasks = tasks

        self.series_num = series.shape[0]
        self.seq_len = series.shape[-1]
        self.task_num = tasks.shape[-1]

    def __len__(self):
        return self.series_num

    def __getitem__(self, idx):
        series = torch.FloatTensor(self.series[idx].astype(float))
        tasks = torch.LongTensor(self.tasks[idx].astype(int))

        return series, tasks


class IDTDataset(Dataset):
    """
        Dataset Class for IDT
    """
    def __init__(self, obs, values, actions, rewards,
                regimes, indices):
        """
            Initialization

            Args:
                obs: multifactor scores(observations)
                    * dtype: np.array(float)
                    * shape:
                        (full_date_num, stock_num, factor_num)
                values: value index series
                    * dtype: np.array(int)
                    * shape: (indices_num, sample_num, period_num)
                actions: action index series
                    * dtype: np.array(int)
                    * shape: (indices_num, sample_num, period_num)
                rewards: reward index series
                    * dtype: np.array(int)
                    * shape: (indices_num, sample_num, period_num)
                regimes: regime series
                    * dtype: np.array(int)
                    * shape: (indices_num, regime_num)
                        * 0: regime up
                        * 1: regime neutral
                        * 2: regime down
                indices: index array
                    * dtype: np.array(int)
                    * shape: (indices_num)
        """
        self.obs = obs
        self.values = values
        self.actions = actions
        self.rewards = rewards
        self.regimes = regimes
        self.indices = indices

        self.date_num, self.sample_num, self.period_num = actions.shape
        self.regime_num = regimes.shape[1]
        self.regime_rng = np.arange(self.regime_num)

        self.regime_idx = defaultdict(np.array)

        for r in self.regime_rng:
            self.regime_idx[r] = np.argwhere(
                regimes[:, r] == 1).ravel()

        self.period_rng = np.arange(self.period_num).reshape(-1, 1)

        assert indices.shape[0] == self.date_num

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
            get item

            Return:
                obs: observations
                    * dtype: torch.FloatTensor
                    * shape:
                        (-1, sample_num*regime_num, period_num, stock_num, factor_num)
                values: values
                    * dtype: torch.LongTensor
                    * shape: (-1, sample_num*regime_num, period_num)
                actions: actions
                    * dtype: torch.LongTensor
                    * shape: (-1, sample_num*regime_num, period_num)
                rewards: rewards
                    * dtype: torch.LongTensor
                    * shape: (-1, sample_num*regime_num, period_num)
        """
        reg = self.regimes[idx].argmax()
        no_regs = list(set(self.regime_rng).difference([reg]))

        indices = [idx]
        # pick regimes
        for r in no_regs:
            picked = np.random.choice(self.regime_idx[r], 1).item()
            indices.append(picked)

        oidx = self.indices[indices].reshape(1, -1) + self.period_rng

        obs = torch.FloatTensor(
            self.obs[oidx].astype(float)).transpose(0, 1).contiguous()
        obs = obs.unsqueeze(1).repeat(
            1, self.sample_num, 1, 1, 1).view(
                self.sample_num*self.regime_num, self.period_num,
                obs.shape[-2], obs.shape[-1]).contiguous()

        values = torch.LongTensor(
            self.values[indices].astype(int)).view(-1, self.period_num)
        actions = torch.LongTensor(
            self.actions[indices].astype(int)).view(-1, self.period_num)
        rewards = torch.LongTensor(
            self.rewards[indices].astype(int)).view(-1, self.period_num)

        return obs, values, actions, rewards


class IPADataset(Dataset):
    """
        Dataset Class for IPA
    """
    def __init__(self, factors, weights):
        """
            Initialization

            Args:
                factors: multifactors for stocks
                    * dtype: np.array
                    * shape: (date_num, stock_num, factor_num)
                weights: target portfolio weights
                    * dtype: np.array
                    * shape: (date_num, stock_num)
        """
        self.factors = factors
        self.weights = weights

        self.stock_num = weights.shape[1]
        self.factor_num = factors.shape[-1]

    def __len__(self):
        return len(self.factors)

    def __getitem__(self, idx):
        """
            get item

            Args:
                idx: index
            Return:
                factors: multifactor scores
                    * dtype: torch.FloatTensor
                    * shape: (-1, stock_num, factor_num)
                weights: target portfolio weights
                    * dtype: torch.FloatTensor
                    * shape: (-1, stock_num, factor_num)
                port_type_idx: portfolio type index
                    * dtype: torch.LongTensor
                    * shape: (-1)

        """
        factors = self.factors[idx]
        weights = self.weights[idx]

        factors = torch.tensor(factors.astype(float)).float()
        weights = torch.tensor(weights.astype(float)).float()
        port_type_idx = torch.tensor(0)

        return factors, weights, port_type_idx


class CrossAssetBERTFinetuningDataset(Dataset):
    """
        Dataset Class for Cross Asset BERT Finetuning
    """
    def __init__(self, factors, up_targets, down_targets):
        """
            Initialization

            Args:
                factors: multifactors for stocks
                    * dtype: np.array
                    * shape: (date_num, stock_num, factor_num)
                up_targets: up targets of stocks
                    * dtype: np.array
                    * shape: (date_num, stock_num)
                down_targets: down targets of stocks
                    * dtype: np.array
                    * shape: (date_num, stock_num)
        """
        self.factors = factors
        self.up_targets = up_targets
        self.down_targets = down_targets

        self.stock_num = self.factors.shape[1]
        self.factor_num = self.factors.shape[-1]
        self.factor_idx = np.arange(self.factor_num)

    def __len__(self):
        return len(self.factors)

    def __getitem__(self, idx):
        """
            getitem

            Args:
                idx: index
            Return:
                factors: multifactors for stocks
                    * dtype: torch.FloatTensor
                    * shape: (-1, stock_num, factor_num)
                up_targets: up targets of stocks
                    * dtype: torch.FloatTensor
                    * shape: (-1, stock_num)
                down_targets: down targets of stocks
                    * dtype: torch.FloatTensor
                    * shape: (-1, stock_num)
        """
        factors_v = self.factors[idx]
        up_targets_v = self.up_targets[idx]
        down_targets_v = self.down_targets[idx]

        factors = torch.FloatTensor(factors_v.astype(float))
        up_targets = torch.FloatTensor(up_targets_v.astype(float))
        down_targets = torch.FloatTensor(down_targets_v.astype(float))

        return factors, up_targets, down_targets


class DIPADataset(Dataset):
    """
        Dataset Class for DIPA
    """
    def __init__(self, enc_factors, factors, returns, data_index,
            trading_period=250):
        """
            Initialization

            Args:
                enc_factors: multifactors for stocks for CrossAssetBERT
                    * dtype: np.array
                    * shape: (date_num, enc_stock_num, factor_num)
                factors: multifactors for stocks
                    * dtype: np.array
                    * shape: (date_num, stock_num, factor_num)
                returns: returns data
                    * dtype: np.array
                    * shape: (date_num, stock_num)
                data_index: data index
                    * dtype: np.array
                    * shape: (data_length)
                trading_period: trading period
                    * default: 250
        """
        self.enc_factors = enc_factors
        self.factors = factors
        self.returns = returns
        self.data_index = data_index
        self.trading_period = trading_period

        self.max_idx = factors.shape[0] - trading_period

        assert data_index.max() <= self.max_idx

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        """
            get item
        """
        idx = self.data_index[idx]
        enc_factors = self.enc_factors[idx:idx+self.trading_period]
        factors = self.factors[idx:idx+self.trading_period]
        returns = self.returns[idx:idx+self.trading_period]

        enc_factors = torch.FloatTensor(enc_factors.astype(float))
        factors = torch.FloatTensor(factors.astype(float))
        returns = torch.FloatTensor(returns.astype(float))

        return enc_factors, factors, returns


class CrossAssetBERTDataset(Dataset):
    """
        Dataset Class for Cross Asset BERT
    """
    def __init__(self, factors, up_targets, down_targets,
                market_targets,
                factor_masking_prob=0.3,
                masking_prob=0.3, masking=-1):
        """
            Initialization

            Args:
                factors: multifactors for stocks
                    * dtype: np.array
                    * shape: (date_num, stock_num, factor_num)
                up_targets: up targets of stocks
                    * dtype: np.array
                    * shape: (date_num, stock_num)
                down_targets: down targets of stocks
                    * dtype: np.array
                    * shape: (date_num, stock_num)
                market_targets: market targets
                    * dtype: np.array
                    * shape: (date_num)
        """
        self.factors = factors
        self.up_targets = up_targets
        self.down_targets = down_targets
        self.market_targets = market_targets.astype(int)

        self.factor_masking_prob = factor_masking_prob
        self.masking_prob = masking_prob

        self.factor_masking_num =\
            int(self.factor_masking_prob * self.factors.shape[-1])

        self.masking = masking
        self.stock_num = self.factors.shape[1]
        self.factor_num = self.factors.shape[-1]
        self.factor_idx = np.arange(self.factor_num)

    def __len__(self):
        return len(self.factors)

    def __getitem__(self, idx):
        """
            getitem

            Args:
                idx: index
            Return:
                factors: multifactors for stocks
                    * dtype: torch.FloatTensor
                    * shape: (-1, stock_num, factor_num)
                up_targets: up targets of stocks
                    * dtype: torch.FloatTensor
                    * shape: (-1, stock_num)
                down_targets: down targets of stocks
                    * dtype: torch.FloatTensor
                    * shape: (-1, stock_num)
                market_targets: market targets
                    * dtype: torch.LongTensor
                    * shape: (-1)
        """
        factors_v = deepcopy(self.factors[idx])
        up_targets_v = self.up_targets[idx]
        down_targets_v = self.down_targets[idx]
        market_targets = self.market_targets[idx]

        if self.masking_prob is not None and\
            np.random.rand() <= self.masking_prob:
            for i in range(self.stock_num):
                mask_idx = np.random.choice(
                    self.factor_idx, self.factor_masking_num,
                    replace=False)
                factors_v[i, mask_idx] = self.masking

        factors = torch.FloatTensor(factors_v.astype(float))
        up_targets = torch.FloatTensor(up_targets_v.astype(float))
        down_targets = torch.FloatTensor(down_targets_v.astype(float))

        return factors, up_targets, down_targets, market_targets


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