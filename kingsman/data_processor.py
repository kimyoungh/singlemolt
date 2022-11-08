"""
    Data Processor for Trading

    @author: Younghyun Kim
    Created on 2022.04.16
"""
from copy import deepcopy
from collections import defaultdict
import numpy as np
import pandas as pd


WINDOW_LIST = [5, 10, 20, 40, 60, 90, 120, 250]


class IndexStrategyProcessor:
    """
        Decision Transformer를 위한
        Investment Strategy Sequence 생성
        * Simulated Annealing 기반
    """
    def __init__(self, returns, fee=0.01,
                reward_list=[-2, 0, +1],
                trading_period=20,
                sample_num=5,
                decay=0.99, eps=1e-6):
        """
            Args:
                returns: returns series
                    * dtype: pd.DataFrame
                    * shape: (date_num, strategy_num)
                fee: trading fee
                    * default: 1%
                reward_list: reward list(worst, neutral, best)
                    * dtype: list
                    * default: [-2, 0, +1]
                trading_period: trading period
                    * default: 20-day
                sample_num: sample number
                    * default: 5
                decay: temperature reduction coefficient
                eps: epsilon
        """
        self.returns = returns
        self.date_num = returns.shape[0]
        self.strategy_num = returns.shape[1]
        self.strategies = returns.columns.values
        self.st_idx = np.arange(self.strategy_num)

        self.fee = fee
        self.reward_list = np.array(reward_list)
        self.trading_period = trading_period
        self.sample_num = sample_num

        self.decay = decay
        self.eps = eps

        if self.date_num < self.trading_period:
            raise Exception("Too short returns series!")

        self.date_length = self.date_num - self.trading_period
        self.max_value = reward_list[2] * trading_period
        self.min_value = reward_list[0] * trading_period

        self.value_table = np.arange(
            self.min_value, self.max_value+ 1)
        self.value_idx = np.arange(len(self.value_table))
        self.reward_idx = np.arange(len(reward_list))

    def generate_overall_dataset(self, pick_num=1, sample_num=None,
                                temp=20000, eps=None):
        """
            generate overall strategy series dataset
            Args:
                pick_num: picking number of strategies for neighbor_op
                sample_num: sample number for each date
                    * default: self.sample_num
                temp: temparature
                eps: epsilon
            Returns:
                dataset: dict of dataset
                    * key: data name
                    * value:
                        st_series: strategy series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        rets_series: return series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        rew_series: rewards series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        val_series: value series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        val_idx_series: value index series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        rew_idx_series: reward index series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        date_series: date series
                            * dtype: np.array
                            * shape: (date_num-trading_period)
        """
        # Best Series
        dataset_best = self.calculate_strategy_series_dataset(
            mode='best', pick_num=pick_num, sample_num=sample_num,
            temp=temp, eps=eps)
        print("best done!")

        # Worst Series
        dataset_worst = self.calculate_strategy_series_dataset(
            mode='worst', pick_num=pick_num, sample_num=sample_num,
            temp=temp, eps=eps)
        print("worst done!")

        # Random Series
        dataset_random = self.calculate_strategy_series_dataset(
            mode='random', pick_num=pick_num, sample_num=sample_num,
            temp=temp, eps=eps)
        print("random done!")

        dataset = defaultdict(np.array)

        dataset['st_series'] = np.concatenate(
            (dataset_best['st_series'], dataset_random['st_series'],
            dataset_worst['st_series']), axis=1)
        dataset['rets_series'] = np.concatenate(
            (dataset_best['rets_series'], dataset_random['rets_series'],
            dataset_worst['rets_series']), axis=1)
        dataset['rew_series'] = np.concatenate(
            (dataset_best['rew_series'], dataset_random['rew_series'],
            dataset_worst['rew_series']), axis=1)
        dataset['val_series'] = np.concatenate(
            (dataset_best['val_series'], dataset_random['val_series'],
            dataset_worst['val_series']), axis=1)
        dataset['val_idx_series'] = np.concatenate(
            (dataset_best['val_idx_series'],
            dataset_random['val_idx_series'],
            dataset_worst['val_idx_series']), axis=1)
        dataset['rew_idx_series'] = np.concatenate(
            (dataset_best['rew_idx_series'],
            dataset_random['rew_idx_series'],
            dataset_worst['rew_idx_series']), axis=1)
        dataset['date_series'] = dataset_best['date_series']

        return dataset

    def calculate_strategy_series_dataset(self, mode='best',
                                        pick_num=1, sample_num=None,
                                        temp=20000, eps=None):
        """
            calculate series of strategy series

            Args:
                mode: Style of strategy series
                    * kind: [best, random, worst]
                    * default: best
                pick_num: picking number of strategies for neighbor_op
                sample_num: sample number for each date
                    * default: self.sample_num
                temp: temparature
                eps: epsilon
            Returns:
                dataset: dict of dataset
                    * key: data name
                    * value:
                        st_series: strategy series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        rets_series: return series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        rew_series: rewards series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        val_series: value series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        val_idx_series: value index series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        rew_idx_series: reward index series
                            * dtype: np.array
                            * shape: (date_num-trading_period, sample_num, trading_period)
                        date_series: date series
                            * dtype: np.array
                            * shape: (date_num-trading_period)
        """
        if sample_num is None:
            sample_num = self.sample_num

        st_series, rets_series,\
            rew_series, val_series,\
                val_idx_series, rew_idx_series = [], [], [], [], [], []

        for i in range(self.date_length):
            st, rets, rews, vals,\
                val_index, rew_index = [], [], [], [], [], []
            for _ in range(sample_num):
                if mode != 'random':
                    st_top, ret_top, rew_top,\
                        val_top, val_idx, rew_idx =\
                        self.search_strategy_series(
                            self.returns.iloc[
                                i+1:i+self.trading_period+1].values,
                            mode=mode, pick_num=pick_num, temp=temp,
                            eps=eps)
                else:
                    st_top, ret_top, rew_top, val_top,\
                        val_idx, rew_idx =\
                        self.random_strategy_series(
                            self.returns.iloc[
                                i+1:i+self.trading_period+1].values)

                st.append(st_top)
                rets.append(ret_top)
                rews.append(rew_top)
                vals.append(val_top)
                val_index.append(val_idx)
                rew_index.append(rew_idx)

            st = np.stack(st, axis=0)
            rets = np.stack(rets, axis=0)
            rews = np.stack(rews, axis=0)
            vals = np.stack(vals, axis=0)
            val_index = np.stack(val_index, axis=0)
            rew_index = np.stack(rew_index, axis=0)

            st_series.append(st)
            rets_series.append(rets)
            rew_series.append(rews)
            val_series.append(vals)
            val_idx_series.append(val_index)
            rew_idx_series.append(rew_index)

        st_series = np.stack(st_series, axis=0)
        rets_series = np.stack(rets_series, axis=0)
        rew_series = np.stack(rew_series, axis=0)
        val_series = np.stack(val_series, axis=0)
        val_idx_series = np.stack(val_idx_series, axis=0)
        rew_idx_series = np.stack(rew_idx_series, axis=0)

        date_series = self.returns.index.values[:self.date_length]

        dataset = defaultdict(np.array)
        dataset['st_series'] = st_series
        dataset['rets_series'] = rets_series
        dataset['rew_series'] = rew_series
        dataset['val_series'] = val_series
        dataset['val_idx_series'] = val_idx_series
        dataset['rew_idx_series'] = rew_idx_series
        dataset['date_series'] = date_series

        return dataset

    def search_strategy_series(self, rets, mode='best', pick_num=1,
                            temp=20000, eps=None):
        """
            Search method for picking strategy series based on mode,
            by Simulated Annealing

            Args:
                rets: returns series
                    * dtype: np.array
                    * shape: (trading_period, strategy_num)
                mode: Style of strategy series
                    * kind: [best, worst]
                    * default: best
                pick_num: picking number of strategies for neighbor_op
                temp: temparature
                eps: epsilon
            Returns:
                series_top: best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                returns_top: return series of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                rewards_top: reward series of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                values_top: value series of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                value_index: value index of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                rewards_index: reward index of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
        """
        if eps is None:
            eps = self.eps

        series_now = self.pick_initial_strategy_series()

        rs, rewards, values = self.calculate_series_values(
            series_now, rets)

        values_now = values

        series_top = series_now
        returns_top = rs
        rewards_top = rewards
        values_top = values

        while temp >= eps:
            series_new = self.neighbor_op(series_now, pick_num)

            returns_new, rewards_new, values_new =\
                self.calculate_series_values(series_new, rets)

            if mode == 'best':
                cost = values_new[0] - values_now[0]
                cost_top = values_new[0] - values_top[0]
            elif mode == 'worst':
                cost = values_now[0] - values_new[0]
                cost_top = values_top[0] - values_new[0]

            if cost > 0:
                series_now = series_new
                values_now = values_new
            else:
                temp_v = cost / temp
                if np.random.uniform(0, 1) < np.exp(temp_v):
                    series_now = series_new
                    values_now = values_new

            if cost_top > 0:
                series_top = series_new
                returns_top = returns_new
                rewards_top = rewards_new
                values_top = values_new

            temp = temp * self.decay

        value_index, rewards_index = [], []

        for i in range(self.trading_period):
            val = values_top[i]
            rew = rewards_top[i]

            val_idx = np.argwhere(self.value_table == val).item()
            rew_idx = np.argwhere(self.reward_list == rew).item()

            value_index.append(val_idx)
            rewards_index.append(rew_idx)

        return series_top, returns_top, rewards_top, values_top,\
            value_index, rewards_index

    def random_strategy_series(self, rets):
        """
            Get Random Strategy Series

            Args:
                rets: returns series
                    * dtype: np.array
                    * shape: (trading_period, strategy_num)
            Returns:
                series_top: best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                returns_top: return series of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                rewards_top: reward series of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                values_top: value series of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                value_index: value index of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                rewards_index: reward index of best strategy series
                    * dtype: np.array
                    * shape: (trading_period)
        """
        series_now = self.pick_initial_strategy_series()

        returns_now, rewards_now, values_now =\
            self.calculate_series_values(series_now, rets)

        value_index, rewards_index = [], []

        for i in range(self.trading_period):
            val = values_now[i]
            rew = rewards_now[i]

            val_idx = np.argwhere(self.value_table == val).item()
            rew_idx = np.argwhere(self.reward_list == rew).item()

            value_index.append(val_idx)
            rewards_index.append(rew_idx)

        return series_now, returns_now, rewards_now, values_now,\
            value_index, rewards_index

    def neighbor_op(self, picked, pick_num=1):
        """
            pick neighbor of picked strategy series

            Args:
                picked: previous picked strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                pick_num: picking number of strategies for neighbor_op
                    * default: 1
        """
        picked_new = deepcopy(picked)
        pick_num = min(pick_num, self.trading_period)

        prng = np.arange(len(picked_new))
        prng = np.random.choice(prng, pick_num, replace=False)

        for p in prng:
            st_p = np.random.choice(self.st_idx, 1).item()
            picked_new[p] = st_p

        return picked_new

    def pick_initial_strategy_series(self):
        """
            Pick Initial Strategy Series
        """
        picked = np.random.choice(self.st_idx,
                                self.trading_period,
                                replace=True)

        return picked

    def calculate_series_values(self, series, returns):
        """
            Calculate Series Values
                * return series, reward series, value series

            Args:
                series: picked strategy series
                    * dtype: np.array
                    * shape: (trading_period)
                returns: return series
                    * dtype: np.array
                    * shape: (trading_period, strategy_num)
            Return:
                rets: strategy return series
                    * dtype: np.array
                    * shape: (trading_period)
                rewards: strategy reward series
                    * dtype: np.array
                    * shape: (trading_period)
                values: strategy value series
                    * dtype: np.array
                    * shape: (trading_period)
        """
        rets = []
        for i, st in enumerate(series):
            if i == 0:
                rets.append(returns[i, st])
            else:
                if st != st_prev:
                    ret = returns[i, st] - (self.fee * 2)
                else:
                    ret = returns[i, st]

                rets.append(ret)

            st_prev = st

        rets = np.array(rets)
        rewards = np.where(rets > 0., self.reward_list[2],
                        np.where(rets == 0, self.reward_list[1],
                                self.reward_list[0]))

        values = rewards[::-1].cumsum()[::-1]

        return rets, rewards, values


class TAProcessor:
    """
        BERTTA 데이터 가공 클래스
    """
    def __init__(self, price, period=250,
                task_periods=np.arange(1, 250),
                eps=1e-6):
        """
            initialization

            Args:
                price: price series
                    * dtype: pd.DataFrame
                    * shape: (date_num, asset_num)
                period: window period
                    * default: 250
                task_periods: periods for task returns
                    * default: np.arange(1, 250)
        """
        self.price = price
        self.period = period
        self.task_periods = task_periods
        self.eps = eps

        self.task_num = len(task_periods)

        self.returns = price.pct_change()
        self.returns.iloc[0] = 0.

        self.date_num, self.asset_num = price.shape
        self.time_length = self.date_num - period + 1

    def processing_dataset(self):
        """
            processing dataset for BERTTA

            Return:
                series_total: total asset time series
                    * dtype: np.array
                    * shape: (time_length, asset_num, period)
                task_answers: task answers
                    * dtype: np.array
                    * shape: (time_length, asset_num, task_num)
                series_index: index of each components of series_total
                    * dtype: np.array
                    * shape: (time_length)
                task_raw: task raw data
                    * dtype: np.array
                    * shape: (time_length, asset_num, task_num)
        """
        series_total, series_index = self.processing_time_series()

        task_answers, task_raws = self.processing_task_answers(
            series_total)

        return series_total, task_answers, series_index, task_raws

    def processing_time_series(self):
        """
            processing time series

            Return:
                series_total: total asset time series
                    * dtype: np.array
                    * shape: (time_length, asset_num, period)
                series_index: index of each components of series_total
                    * dtype: np.array
                    * shape: (time_length)
        """
        series_total = []
        for t in range(self.time_length):
            series = self.returns.iloc[t:self.period+t]
            series = (1 + series).cumprod()
            series = self.h_transform(series).values

            series_total.append(series.transpose())

        series_total = np.stack(series_total, axis=0)
        series_index = self.returns.index[self.period-1:].values

        return series_total, series_index

    def processing_task_answers(self, series_total):
        """
            processing task answers(price returns)

            Args:
                series_total: total asset time series
                    * dtype: np.array
                    * shape: (time_length, asset_num, period)
            Return:
                task_answers: task answers
                    * dtype: np.array
                    * shape: (time_length, asset_num, task_num)
                task_raw: task raw data
                    * dtype: np.array
                    * shape: (time_length, asset_num, task_num)
        """
        task_raw = []
        task_answers = []
        for p in self.task_periods:
            diffs = series_total[:, :, -1] - series_total[:, :, -1-p]
            answers = (diffs > 0).astype(int)

            task_answers.append(answers)
            task_raw.append(diffs)

        task_answers = np.stack(task_answers, axis=2)
        task_raw = np.stack(task_raw, axis=2)

        return task_answers, task_raw

    def h_transform(self, x_in):
        """
            Apply H Transform

            Args:
                x_in: price series
                    * shape: (date_num, asset_num)
            Return:
                x_out: h transformed price series
                    * shape: (date_num, asset_num)
        """
        x_out = np.sign(x_in) * (
            np.sqrt(np.abs(x_in) + 1) - 1) + (self.eps*x_in)
        return x_out


class PortfolioGenerator:
    """
        Portfolio Generator Class
        조건에 맞는 포트폴리오 생성
    """
    def __init__(self, returns, port_prob=0.1,
                window_list=None, decay=0.99, eps=1e-6):
        """
            Args:
                returns: returns series
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                port_prob: number prob of generated portfolio
                    * default: 10%
                window_list: 성과 계산에 필요한 향후 시점 리스트
                    * dtype: list
                decay: temperature reduction coefficient
        """
        self.returns = returns
        self.date_num = returns.shape[0]
        self.stock_num = returns.shape[1]

        if window_list is None:
            window_list = WINDOW_LIST
        self.window_list = window_list
        self.window_num = len(self.window_list)
        self.max_window = max(window_list)

        if self.date_num < (self.max_window + 1):
            raise Exception("Too short returns series!")

        self.limit_date_idx = self.date_num - self.max_window

        self.port_prob = port_prob

        self.port_stock_num = round(self.port_prob * self.stock_num)

        self.stock_index = np.arange(self.stock_num)

        self.decay = decay
        self.eps = eps

    def generate_portfolio_series(self, returns, pick_num=1,
                                temp=20000, eps=None):
        """
            Search method for picking best portfolio
            by simulated annealing

            Args:
                returns: returns series of overall stocks
                    * dtype: pd.DataFrame or
                    * shape: (target_date_num, stock_num)
                pick_num: picking number of stocks for neighbor_op
                    * default: 1
                temp: temperature
        """
        if eps is None:
            eps = self.eps

        values = []

        weights = np.zeros((self.limit_date_idx, self.stock_num))
        for i in range(self.limit_date_idx):
            top_idx, value_top, _ = self.search_best_portfolio(
                returns.iloc[i:], pick_num, temp, eps)
            weights[i, top_idx] = 1. / self.port_stock_num

            values.append(value_top)
            print(returns.index[i], end='\r')

        weights = pd.DataFrame(weights, columns=returns.columns,
                            index=returns.index[:self.limit_date_idx])

        values = pd.Series(values,
            index=returns.index[:self.limit_date_idx])

        return weights, values

    def search_best_portfolio(self, returns, pick_num=1,
                            temp=20000, eps=None):
        """
            Search method for picking best portfolio
            by simulated annealing

            Args:
                returns: returns series of overall stocks
                    * dtype: pd.DataFrame or np.array
                    * shape: (target_date_num, stock_num)
                pick_num: picking number of stocks for neighbor_op
                    * default: 1
                temp: temperature
        """
        assert returns.shape[0] >= (self.max_window + 1)

        if eps is None:
            eps = self.eps

        if isinstance(returns, pd.DataFrame):
            rets = deepcopy(returns.values)
        elif isinstance(returns, np.array):
            rets = deepcopy(returns)

        port_now = self.pick_initial_portfolio_stocks()
        port_top = port_now
        rets_now = rets[:, port_now]
        rets_top = rets_now

        value_now = self.calc_port_value(rets_now)
        value_top = value_now

        while temp >= eps:
            port_new = self.neighbor_op(port_now, pick_num)
            rets_new = rets[:, port_new]

            value_new = self.calc_port_value(rets_new)

            cost = value_new - value_now
            cost_top = value_new - value_top

            if cost > 0:
                port_now = port_new
                value_now = value_new
                rets_now = rets_new
            else:
                temp_v = cost / temp
                if np.random.uniform(0, 1) < np.exp(temp_v):
                    port_now = port_new
                    value_now = value_new
                    rets_now = rets_new

            if cost_top > 0:
                port_top = port_new
                value_top = value_new
                rets_top = rets_new

            temp = temp * self.decay

        cost_top = value_now - value_top

        if cost_top > 0:
            port_top = port_now
            value_top = value_now
            rets_top = rets_now

        return port_top, value_top, rets_top

    def calc_port_value(self, returns):
        """
            calculate portfolio value

            Args:
                returns: stock returns series
                    * dtype: np.array
                    * shape: (date_num, stock_num)
            Return:
                value: portfolio value
        """
        value = 0.
        for window in self.window_list:
            val = self.calc_mret_to_lpm(
                returns[1:window+1]
            )

            value += val / self.window_num

        return value

    def neighbor_op(self, picked, pick_num=1):
        """
            pick neighbor or picked stocks

            Args:
                picked: previous picked stocks
                    * dtype: np.array
                    * shape: (port_stock_num)
                pick_num: picking number of stocks for neighbor_op
                    * default: 1
        """
        picked_new = deepcopy(picked)
        stocks_out = np.setdiff1d(self.stock_index, picked_new)

        pick_num = min(self.port_stock_num, pick_num)

        prng = np.arange(picked_new.shape[0])
        prng = np.random.choice(prng, pick_num, replace=False)

        new_picked = np.random.choice(stocks_out,
                                    pick_num, replace=False)

        picked_new[prng] = new_picked
        picked_new.sort()

        return picked_new

    def pick_initial_portfolio_stocks(self):
        """
            Pick initial portfolio stocks
        """
        picked = np.random.choice(self.stock_index,
                    self.port_stock_num, replace=False)
        picked.sort()

        return picked

    def calc_mret_to_lpm(self, port_returns):
        """
            calculate mret / lpm

            Args:
                port_returns: portfolio returns
                    * dtype: pd.DataFrame or np.array
                    * shape: (forward_date_num, port_stock_num)
            Return:
                mret_to_lpm: mean return
        """
        mret = self.calc_mean_returns(port_returns)
        lpm = self.calc_lpm(port_returns)

        mret_to_lpm = mret - lpm

        return mret_to_lpm

    def calc_mean_returns(self, port_returns):
        """
            향후 기간 포트폴리오 평균 수익률

            Args:
                port_returns: portfolio returns
                    * dtype: pd.DataFrame or np.array
                    * shape: (forward_date_num, port_stock_num)
            Return:
                mret: mean return
        """
        if isinstance(port_returns, pd.DataFrame):
            port_returns = port_returns.values

        init_weights = np.ones(self.port_stock_num) / self.port_stock_num

        weights = np.zeros_like(port_returns)
        weights[0] = init_weights

        for i in range(weights.shape[0] - 1):
            weights_rec = weights[i] * (1 + port_returns[i])
            weights_rec = weights_rec / weights_rec.sum()

            weights[i + 1] = weights_rec

        port_rets = (weights * port_returns).sum(axis=1)

        mret = port_rets.mean()

        return mret

    def calc_lpm(self, port_returns):
        """
            향후 기간 포트폴리오 Lower Partial Moment

            Args:
                port_returns: portfolio returns
                    * dtype: pd.DataFrame or np.array
                    * shape: (forward_date_num, port_stock_num)
            Return:
                lpm: lpm
        """
        if isinstance(port_returns, pd.DataFrame):
            port_returns = port_returns.values

        init_weights = np.ones(self.port_stock_num) / self.port_stock_num

        weights = np.zeros_like(port_returns)
        weights[0] = init_weights

        for i in range(weights.shape[0] - 1):
            weights_rec = weights[i] * (1 + port_returns[i])
            weights_rec = weights_rec / weights_rec.sum()

            weights[i + 1] = weights_rec

        port_rets = (weights * port_returns).sum(axis=1)
        lpm = np.where(port_rets < 0, port_rets, 0)
        lpm = -lpm.mean()

        return lpm


class PriceFactorProcessor:
    """
        Data Processor for Stocks Price Factors
    """
    def __init__(self, returns_data: pd.DataFrame = None,
                window_list=None,
                eps: float = 1e-6):
        """
            Initialization

            Args:
                returns_data: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                eps: epsilon
        """
        self.returns_data = returns_data

        if window_list is None:
            self.window_list = WINDOW_LIST

        self.price_points = np.array([-250, -120, -90, -60,
                                    -40, -20, -10, -5, -1])
        self.eps = eps

    def calculate_mf(self, returns_data: pd.DataFrame = None,
                    normalize=True):
        """
            calculate mf

            Args:
                returns_data: returns data
                    * dtype: pd.DataFrame
                    * shape: ( date_num, stock_num)
        """
        if returns_data is None:
            returns_data = self.returns_data

        factors = pd.DataFrame()
        for w in self.window_list:
            pm = self._calc_price_momentum(returns_data, window=w)
            pm = pm.stack(dropna=False)
            factors = pd.concat((
                factors, pm.to_frame('pm_'+str(w))), axis=1)

            rsi = self._calc_rsi(returns_data, window=w)
            rsi = rsi.stack(dropna=False)
            factors = pd.concat((
                factors, rsi.to_frame('rsi_'+str(w))), axis=1)

            vol = self._calc_volatility(returns_data, window=w)
            vol = vol.stack(dropna=False)
            factors = pd.concat((
                factors, vol.to_frame('vol_'+str(w))), axis=1)

            lpm = self._calc_lpm(returns_data, window=w)
            lpm = lpm.stack(dropna=False)
            factors = pd.concat((
                factors, lpm.to_frame('lpm_'+str(w))), axis=1)

        factors = factors.reset_index()

        returns_index = returns_data.index

        factors = factors.rename(
            columns={'level_0': 'trade_date', 'level_1': 'code'})

        factors = factors[
            factors['trade_date'] >= returns_index[
                self.window_list[-1] - 1]]
        factors = factors.where(pd.notnull(factors), None)

        if normalize:
            factors = self.minmax_scaling(factors)

        factors_v = factors.values[:, 2:].reshape(
            -1, returns_data.shape[1], factors.shape[-1] - 2)
        factors_index = factors.values[:, :2].reshape(
            -1, returns_data.shape[1], 2)

        # Calculate Price Point Data
        price_series = (1 + returns_data).cumprod()
        price_data, _ = self._calc_price_point(
            price_series, normalize=normalize)

        assert factors_v.shape[:2] == price_data.shape[:2]

        factors_v = np.concatenate(
            (factors_v, price_data), axis=-1)

        return factors_v, factors_index

    def minmax_scaling(self, factors):
        """
            minmax scaling
        """
        dates = factors['trade_date'].unique()
        codes = factors['code'].unique()

        factors_v = factors.values
        factors_index_v = factors.values[:, :2]
        factors_v = factors_v[:, 2:].reshape(
            dates.shape[0], codes.shape[0], -1)

        fmax = factors_v.max(1, keepdims=True)
        fmin = factors_v.min(1, keepdims=True)

        normalized = (factors_v - fmin) / (fmax - fmin + self.eps)

        normalized = normalized.reshape(-1, normalized.shape[-1])

        normalized = np.concatenate(
            (factors_index_v, normalized), axis=1)

        normalized = pd.DataFrame(normalized, columns=factors.columns)
        normalized = normalized.where(pd.notnull(normalized), -1)

        return normalized

    def _calc_price_point(self, price_series, normalize=True):
        """
            calculate price point series
        """
        max_window = max(abs(self.price_points))
        length, stock_num = price_series.shape

        price_data = []
        price_index_data = []

        for i in range(length - max_window + 1):
            pdata = price_series.iloc[
                i:i+max_window].transpose().iloc[:, self.price_points]
            price_data.append(pdata.values)
            price_index_data.append(
                list(zip(
                    [price_series.index[i+max_window-1]] * stock_num,
                    pdata.index.values)))

        price_data = np.stack(price_data, axis=0)
        price_index_data = np.stack(price_index_data, axis=0)

        if normalize:
            pmax = price_data.max(-1, keepdims=True)
            pmin = price_data.min(-1, keepdims=True)

            price_data = (price_data - pmin) / (pmax - pmin + self.eps)

        return price_data, price_index_data

    def _calc_price_momentum(self, returns, window=5, log=False):
        """
            calculate price returns

            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: moving window
                log: 로그수익률 여부
            Return:
                pmom: price momentums
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
        """
        if not log:
            rets = np.log(returns + 1.)
        else:
            rets = returns.copy()

        pmom = rets.rolling(window, min_periods=window).sum()
        pmom = np.exp(pmom) - 1.

        return pmom

    def _calc_rsi(self, returns, window=5):
        """
            calculate relative strength indicators

            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: moving window
            Return:
                rsi: rsi
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
        """
        au = (returns > 0).astype(float).rolling(
            window, min_periods=window).mean()
        ad = (returns <= 0).astype(float).rolling(
            window, min_periods=window).mean()

        rsi = au / (au + ad)

        return rsi

    def _calc_volatility(self, returns, window=5):
        """
            calculate volatility

            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: moving window
            Return:
                volatility
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
        """
        volatility = returns.rolling(window, min_periods=window).std()

        return volatility

    def _calc_skew(self, returns, window=5):
        """
            calculate skewness
            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: moving window
            Return:
                skew
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
        """
        skew = returns.rolling(window, min_periods=window).skew()

        return skew

    def _calc_lpm(self, returns, window=5, tau=0.):
        """
            calculate lower partial moment

            Args:
                returns: returns data
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
                window: moving window
                tau: return level criterion
                    * default: 0.
            Return:
                lpm
                    * dtype: pd.DataFrame
                    * shape: (date_num, stock_num)
        """
        lpm = tau - returns
        lpm = lpm.where(lpm > 0, 0).rolling(
            window, min_periods=window).mean()

        return lpm

    def _calc_p_up(self, rolling_returns):
        """
            calculate probability of up-returns of all stocks

            Args:
                rolling_returns: pd.DataFrame(date_num, 1)
        """
        ups = (rolling_returns > 0.).astype(float).mean(1)
        ups = pd.DataFrame(ups, columns=['p_up'])

        return ups


class OHLCVDataProcessor:
    """
        Data Processor for Trading
    """
    def __init__(self, ohlcv_data: dict = None,
                eps: float = 1e-6, translate_cols: bool = False):
        """
            Initialization

            Args:
                ohlcv_data: dict of stock candle data series
                    * dtype: dict
                    * key: asset code
                    * value: stock candle data series
                        * dtype: pd.DataFrame
                        * shape: (date_num, factor_num)
                eps: tiny number
                    * default: 1e-6
                translate_cols: translate cols kor -> eng
                    * default: False

                데이터 정규화는 dataset에서 실시
                1) price horizontal,
                2) price cross-sectional,
                3) return horizontal,
                4) return cross-sectional
        """
        self.col_translation_kr = {
            '시가': 'open',
            '고가': 'high',
            '저가': 'low',
            '종가': 'close',
            '거래량': 'volume',
            '거래대금': 'value'
        }
        self.cols = ['open', 'high', 'low', 'close', 'value']
        self.cols_num = len(self.cols)
        self.feature_num = 4 * self.cols_num

        self.eps = eps

        for key, value in ohlcv_data.items():
            if translate_cols:
                value = value.rename(
                    columns=self.col_translation_kr)
            value.index.rename('date', inplace=True)
            value = value[self.cols]

            ohlcv_data[key] = value

        self.ohlcv_data = ohlcv_data
        self.assets = list(ohlcv_data.keys())
        self.asset_num = len(self.assets)

    def calc_returns(self):
        """
            calculate returns
        """
        price = pd.concat((value['close'].rename(key)
                    for key, value in self.ohlcv_data.items()), axis=1)

        returns = price.pct_change().shift(-1).dropna()

        return returns

    def calc_feature_data(self):
        """
            calculate asset feature data

            Return:
                features: feature data of all assets
                    * dtype: np.array
                    * shape: (asset_num, date_num, feature_num)
                dates: date series
                    * dtype: np.array
                    * shape: (date_num)
        """
        features = []
        dates = None
        for key, value in self.ohlcv_data.items():
            value = np.log(value + self.eps)

            returns = value.diff(1).iloc[1:]
            value = value.iloc[1:]

            feat = pd.concat((value, value, returns, returns),
                            axis=1).values

            features.append(feat)

            if dates is None:
                dates = value.index.values

        features = np.array(
            features).reshape(self.asset_num, -1, self.feature_num)

        return features, dates

    def normalize_features(self, features):
        """
            normalize features

            Args:
                features: feature data
                    * dtype: np.array
                    * shape: (asset_num, date_num, feature_num)
        """
        num = self.cols_num
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


class StrategyProcessor:
    """
        Strategy Processor for Learning Trading

        추출 전략:
            1) 단위 시점 최대 수익률 전략
            2) 임의 전략 이후 시점 단위 시점 최대 수익률 전략
    """
    def __init__(self, returns, st_weights, fee=0.004,
                initial_t=0):
        """
            Initialization

            Args:
                returns: asset returns
                    * dtype: pd.DataFrame
                    * shape: (date_num, asset_num)
                    * 익일 수익률
                st_weights: strategy weights based on each time
                    * dtype: dict
                    * keys: strategies
                    * values: strategy weights
                        * dtype: pd.DataFrame
                        * shape: (date_num, asset_num)
                fee: trading_fee
                    * default: 0.004
                initial_t: initial time position #
                    * default: 0
                    * 관찰 데이터를 고려하여, 관찰 데이터 마지막 시점 번호를 입력하여
                    위치를 맞추기 위함
        """
        # 데이터 크기 확인
        for _, value in st_weights.items():
            assert returns.shape == value.shape

        self.returns = returns
        self.st_weights = st_weights
        self.fee = fee
        self.initial_t = initial_t

        self.strategies  = ['buy_and_hold'] + list(st_weights.keys())
        self.bah_idx = 0

        self.st_returns = self.calc_strategy_returns()

    def calc_daily_best_strategies(self, mode='max'):
        """
            calculate daily best strategies

            Args:
                mode: 수익률 목표
                    'max': max returns
                    'min': min returns
            Return:
                best_strategies: best strategy index list
                    * dtype: pd.DataFrame
                    * shape: (date_num, 1)
                pos_list: time position list
                    * dtype: np.array
                    * shape: (date_num)
        """
        best_strategies = []

        for i in range(self.returns.shape[0]):
            if mode == 'max':
                idx = self.st_returns.iloc[i].values.argmax() + 1
            elif mode == 'min':
                idx = self.st_returns.iloc[i].values.argmin() + 1

            best_strategies.append(idx)

        pos_list = np.arange(self.returns.shape[0]) + self.initial_t

        assert len(pos_list) == len(best_strategies)

        return np.array(best_strategies), pos_list

    def calc_daily_best_strategies_rebalanced(self, mode='max'):
        """
            calculate daily best strategies rebalanced

            Args:
                mode: 수익률 목표
                    'max': max returns
                    'min': min returns
            Return:
                st_series: 2-day strategies series
                    * dtype: np.array
                    * shape: (date_num - 1, strategy_num - 1, 2)
                time_pos: 2-day data time position #
                    * dtype: np.array
                    * shape: (date_num - 1, strategy_num - 1, 2)
        """
        st_series = []
        time_pos = []

        r_length = self.returns.shape[0]

        for i in range(r_length - 1):
            ret_i = self.returns.iloc[i+1].values
            for s_idx, strategy in enumerate(self.strategies[1:]):
                st_rets = []

                w_rec = self.calc_recent_weights(
                    self.st_weights[strategy].iloc[i].values,
                    self.returns.iloc[i].values)

                bah_ret = (w_rec * ret_i).sum()
                st_rets.append(bah_ret)

                for _, next_st in enumerate(self.strategies[1:]):
                    w_next = self.st_weights[next_st].iloc[i+1].values
                    wdiff = abs(w_next - w_rec).sum()

                    ret = (w_next * ret_i).sum() - (wdiff * self.fee)

                    st_rets.append(ret)

                if mode == 'max':
                    best_idx = np.argmax(st_rets)
                elif mode == 'min':
                    best_idx = np.argmin(st_rets)

                st_series.append([s_idx + 1, best_idx])
                time_pos.append([i, i + 1])

        st_series = np.array(st_series).astype(int)
        time_pos = np.array(time_pos).astype(int)

        st_series = st_series.reshape(r_length - 1, -1, 2)
        time_pos = time_pos.reshape(r_length - 1, -1, 2) + self.initial_t

        return st_series, time_pos

    def calc_strategy_returns(self):
        """
            calculate strategy returns
        """
        st_returns = pd.DataFrame(index=self.returns.index)

        for key, value in self.st_weights.items():
            rets = (self.returns * value).sum(1)

            st_returns[key] = rets

        return st_returns

    def calc_recent_weights(self, weights, rets):
        """
            calculate recent weights

            Args:
                weights: asset weights
                    * dtype: np.array
                    * shape: (asset_num)
                rets: asset returns
                    * dtype: np.array
                    *shape: (asset_num)
        """
        weights_rec = weights * (1. + rets)
        weights_rec = weights_rec / weights_rec.sum()

        return weights_rec