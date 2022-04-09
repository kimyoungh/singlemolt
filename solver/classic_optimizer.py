"""
    최적화 비중을 계산해주는 모듈

    @author: Younghyun Kim
    Created on 2021.10.05
"""
import numpy as np
import pandas as pd
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer


class ClassicOptimizer:
    """
        Classic Optimizer
    """
    def __init__(self, m=100,
                buying_fee=0.01, selling_fee=0.01,
                min_cash_rate=0.01):
        """
            Initialization

            Args:
                m: big number
        """
        self.m = m
        self.buying_fee = buying_fee
        self.selling_fee = selling_fee
        self.min_cash_rate = min_cash_rate

    def max_sr(self, returns, nonneg=True, adjust=True):
        """
            Maximize Sharpe Ratio

            Args:
                returns: pd.DataFrame or np.array
            Return:
                weights: np.array(N) 
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.values

        creturns = returns * self.m
        cov = np.cov(creturns.transpose())
        cov = np.nan_to_num(cov)

        mu = creturns.mean(0).reshape(-1)
        mu_min = abs(mu.min())
        if mu[mu > 0].shape[0] == 0:
            mu += mu_min
        mu = np.nan_to_num(mu)

        weights = cp.Variable(returns.shape[1])
        cov_cp = cp.Parameter((cov.shape[1], cov.shape[0]), symmetric=True)

        objective = cp.Minimize(cp.sum_squares(cov_cp @ weights))

        constraints = [mu.T @ weights >= 1]
        if nonneg:
            constraints.append(0 <= weights)
        
        prob = cp.Problem(objective, constraints)
        assert prob.is_dpp()

        cov = torch.FloatTensor(cov.astype(float))

        cvxpylayer = CvxpyLayer(prob, parameters=[cov_cp],
                                variables=[weights])
        
        weights, = cvxpylayer(cov)

        if adjust:
            weights = self.adjust_weights(weights)

        return weights.numpy()

    def min_var(self, returns):
        """
            Minimum Variance Portfolio
            Args:
                returns: pd.DataFrame or np.array
            Return:
                weights: np.array(N)
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.values

        creturns = returns * self.m
        cov = np.cov(creturns.transpose())
        cov = np.nan_to_num(cov)

        weights = cp.Variable(returns.shape[1])
        cov_cp = cp.Parameter((cov.shape[1], cov.shape[0]),
                            symmetric=True)

        objective = cp.Minimize(cp.sum_squares(cov_cp @ weights))

        constraints = [cp.sum(weights) == 1, 0 <= weights]

        prob = cp.Problem(objective, constraints)
        assert prob.is_dpp()

        cov = torch.FloatTensor(cov.astype(float))

        cvxpylayer = CvxpyLayer(prob, parameters=[cov_cp],
                                variables=[weights])

        weights, = cvxpylayer(cov)

        return weights.numpy()

    def max_div(self, returns, nonneg=True, adjust=True):
        """
            Maximum Diversification Portfolio

            Args:
                returns: pd.DataFrame or np.array
            Return:
                weights: np.array(N)
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.values

        creturns = returns * self.m
        cov = np.cov(creturns.transpose())
        cov = np.nan_to_num(cov)

        sig = creturns.std(0).reshape(-1)
        sig = np.nan_to_num(sig)

        weights = cp.Variable(returns.shape[1])
        cov_cp = cp.Parameter((cov.shape[1], cov.shape[0]),
                                symmetric=True)

        objective = cp.Minimize(cp.sum_squares(cov_cp @ weights))

        constraints = [sig.T @ weights >= 1]
        if nonneg:
            constraints.append(0 <= weights)

        prob = cp.Problem(objective, constraints)
        assert prob.is_dpp()

        cov = torch.FloatTensor(cov.astype(float))

        cvxpylayer = CvxpyLayer(prob, parameters=[cov_cp],
                                variables=[weights])

        weights, = cvxpylayer(cov)

        if adjust:
            weights = self.adjust_weights(weights)

        return weights.numpy()

    def mv_mean(self, returns):
        """
            Mean-Variance Portfolio with min ret based on mean ret
            Args:
                returns: pd.DataFrame or np.array
            Return:
                weights: np.array(N)
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.values

        creturns = returns * self.m
        cov = np.cov(creturns.transpose())
        cov = np.nan_to_num(cov)

        weights = cp.Variable(returns.shape[1])
        cov_cp = cp.Parameter((cov.shape[1], cov.shape[0]),
                            symmetric=True)

        mu = creturns.mean(0).reshape(-1)
        mu_min = abs(mu.min())
        if mu[mu > 0].shape[0] == 0:
            mu += mu_min
        mu = np.nan_to_num(mu)
        mret = mu.mean().item()

        objective = cp.Minimize(cp.sum_squares(cov_cp @ weights))

        constraints = [cp.sum(weights) == 1,
                        mu.T @ weights >= mret,
                        0 <= weights]

        prob = cp.Problem(objective, constraints)
        assert prob.is_dpp()

        cov = torch.FloatTensor(cov.astype(float))

        cvxpylayer = CvxpyLayer(prob, parameters=[cov_cp],
                                variables=[weights])

        weights, = cvxpylayer(cov)

        return weights.numpy()

    def pm_port(self, returns, topk=5, return_type='pct'):
        """
            Price Momentum Equal Weight Portfolio with TopK
            Args:
                returns: pd.DataFrame or np.array
                topk: top K
                return_type: return type(log or pct)
            Return:
                weights: np.array(N)
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.values

        if return_type == 'pct':
            returns = np.log(returns + 1.)

        crets = returns.sum(0)
        crets = np.nan_to_num(crets)

        crank = crets.argsort()
        weights = np.zeros(returns.shape[1])
        weights[crank[-topk:]] = 1. / topk

        return weights

    def lowvol_port(self, returns, topk=5):
        """
            Lowvol Equal Weight Portfolio with TopK
            Args:
                returns: pd.DataFrame or np.array
                topk: top K
            Return:
                weights: np.array(N)
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.values

        sig = returns.std(0)
        sig = np.nan_to_num(sig)

        srank = sig.argsort()
        weights = np.zeros(returns.shape[1])
        weights[srank[:topk]] = 1. / topk

        return weights

    def ew_port(self, n):
        """
            Equal Weight Portfolio with n assets
            Args:
                n: asset num
            Return:
                weights: np.array(n)
        """
        weights = torch.ones(n) / n

        return weights

    def solve_amount(self, asset_prices, asset_embs, optimal_emb, wealth):
        """
            Solving method for trading amounts

            Args:
                asset_prices: np.array 수량 계산에 필요한 자산 별 가격(1 X N)
                asset_embs = np.array 자산 별 임베딩(N X M)
                optimal_emb: 최적 포트폴리오 임베딩(1 X M)
                wealth: 총 투자금

            Return:
                buying_amount: 종목 별 수량
                prob_value: 최적과 최종 포트폴리오 거리(L2)
        """
        wealth =\
            wealth * (1. - max(self.buying_fee, self.selling_fee))  # 비용 고려
        wealth = wealth * (1. - self.min_cash_rate)  # 최소 보유 현금 고려
        asset_embs_v = asset_embs.transpose() * asset_prices / wealth
        asset_prices = asset_prices.reshape(-1)

        buying_amount = cp.Variable(asset_prices.shape[0])
        optimal_emb = optimal_emb.reshape(-1)

        objective = cp.Minimize(self.m *
                                cp.sum_squares((asset_embs_v @ buying_amount)
                                                - optimal_emb))
        constraints = [buying_amount >= 0,
                    asset_prices.T @ buying_amount == wealth]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        buying_amount = np.round(buying_amount.value, 0)

        return buying_amount, prob.value

    def get_replicated_buying_amounts(self, closes, asset_embs, weights,
                                    insts=['A069500', 'A229200',
                                            'A114800', 'A251340'],
                                    topk=10, wealth=50000000):
        """
            closes: pd.Series 종목 별 종가(stock_num)
            asset_embs: torch.tensor 종목 별 임베딩(1, stock_num, emb_dim)
            weights: torch.tensor 종목 별 투자비중(1, stock_num)
            insts: list 복제에 활용될 시장 ETF(default: K200, KQ150)
            topk: 복제하기 위한 상위 종목 수

            * closes, asset_embs, weights는 종목 별 순서가 일치해야함

            Return:
                amounts: pd.DataFrame 매수수량
                aweights: pd.DataFrame 매수수량을 바탕으로 한 투자비중
                value_est: closes를 바탕으로 계산한 총금액
                prob_value: 임베딩 거리
        """
        ins = []
        for inst in insts:
            ind = np.argwhere(closes.index == inst).item()
            ins.append(ind)

        ranks = weights.argsort(descending=True)
        ranks = ranks.cpu().numpy().reshape(-1)
        sel = np.unique(np.concatenate((ranks[:topk], ins), axis=-1))

        optimal_emb = self.calc_optimal_emb(asset_embs, weights)

        embs = asset_embs[0, sel].cpu().numpy()
        optimal_emb = optimal_emb.view(-1, 1).cpu().numpy()

        amounts, prob_value = self.solve_amount(closes.iloc[sel].values,
                                        embs, optimal_emb, wealth)

        amounts = pd.DataFrame(amounts.reshape(-1, 1),
                            index=closes.index[sel],
                            columns=['amounts'])
        amounts = amounts[amounts['amounts'] > 0]

        closes = pd.DataFrame(closes.values, index=closes.index, columns=amounts.columns)

        value_est = (amounts.values.ravel() * closes.loc[amounts.index].values.ravel()).sum()

        aweights = (amounts * closes.loc[amounts.index]) / value_est

        return amounts, aweights, value_est, prob_value

    def calc_optimal_emb(self, asset_embs, weights):
        """
            calculate optimal embedding
            Args:
                asset_embs: torch.tensor
                    (batch_size, stock_num, emb_dim)
                weights: torch.tensor
                    (batch_size, stock_num)
        """
        optimal_emb = torch.matmul(weights, asset_embs)

        return optimal_emb

    def adjust_weights(self, weights):
        """
            비중 조정
                * nonneg일때,
                    weights /= weights.sum()
                * weights[weights > 0].sum() > 0 일때,
                    weights /= weights[weights > 0].sum()
                * weights[weights > 0].sum() < 0이고,
                weights[weights < 0] != 0일때,
                    weights /= -weights[weights < 0].sum()
        """
        if (weights != 0).sum() > 0:
            weights = weights / abs(weights).max()

        wpos_sum = weights[weights > 0].sum()
        wneg_sum = -weights[weights < 0].sum()

        if weights.sum() != 0:
            weights /= max(wpos_sum, wneg_sum)
        
        return weights