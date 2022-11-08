"""
    Module for stock Transformers

    @author: Younghyun Kim
    Created on 2022.04.10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from kingsman.cfg.knights_configs\
    import (FT_CROSS_ASSET_BERT_CONFIG, CROSS_ASSET_BERT_CONFIG,
        IPA_CONFIG, TRADING_BERT_CONFIG, GLEN_SCOTIA_CONFIG,
        REBAL_CONFIG, IDT_CONFIG, BERTTA_CONFIG, TT_CONFIG)
from kingsman.layers import TransformerEnc, TransformerDec, Mapping


class TradingTransformer(nn.Module):
    """
        Trading Transformer
    """
    def __init__(self, config: dict = None):
        """
            Initialization

            Args:
                config: config file
                    * dtype: dict
        """
        super().__init__()

        if config is None:
            config = TT_CONFIG
        self.config = config

        self.factor_num = config['factor_num']
        self.d_model = config['d_model']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.nhead = config['nhead']
        self.nlayers = config['nlayers']
        self.activation = config['activation']
        self.asset_map_nlayers = config['asset_map_nlayers']
        self.pp_map_nlayers = config['pp_map_nlayers']

        # Trading Position Encodings
        ## 0: Zero Position, 1: Long Position
        self.tp_embeds = nn.Embedding(2, self.d_model)

        # Asset Embedding
        self.asset_embeds = Mapping(self.factor_num, self.d_model,
                                self.asset_map_nlayers, 'first',
                                self.slope, self.dropout, True)

        # Transformer
        self.attn = TransformerEnc(
            self.d_model, self.nhead, self.nlayers,
            self.d_model * 2, self.dropout,
            self.activation, True)

        # Position Predictor
        self.position_preds = Mapping(self.d_model, 1,
                self.pp_map_nlayers, 'last',
                self.slope, self.dropout,
                False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        " Initialize Model Weights "
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, asset_in, trading_position=None,
                softmax=False):
        """
            Inference

            Args:
                asset_in: asset inputs
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num, factor_num)
                trading_position: trading position
                    * dtype: torch.LongTensor
                    * shape: (batch_size, asset_num)
                    * value
                        * 0: Zero Position
                        * 1: Long Position
                softmax: apply softmax or not to preds
            Returns:
                preds: position prediction
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num)
        """
        batch_size, asset_num, _ = asset_in.shape

        assets = self.asset_embeds(asset_in)

        if trading_position is None:
            trading_position = torch.zeros(
                (batch_size, asset_num)).type(torch.long).to(self.device)
        trading_p = self.tp_embeds(trading_position)

        embeds = assets + trading_p

        attn_out = self.attn(embeds, seq_mask=False,
                            src_key_padding_mask=None)

        preds = self.position_preds(attn_out).squeeze(-1)

        if softmax:
            preds = preds.softmax(dim=-1)

        return preds


class InvestingDecisionTransformer(nn.Module):
    """
        Investing Decision Transformer
    """
    def __init__(self, config: dict = None):
        """
            Initialization

            Args:
                config: config file
                    * dtype: dict
        """
        super().__init__()

        if config is None:
            config = IDT_CONFIG
        self.config = config

        self.factor_num = config['factor_num']
        self.d_model = config['d_model']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.nhead = config['idt_nhead']
        self.nlayers = config['idt_nlayers']
        self.activation = config['activation']
        self.max_len = config['max_len']
        self.actions = config['actions']
        self.values = config['values']
        self.rewards = config['rewards']
        self.K = config['K']
        self.action_num = len(self.actions)
        self.value_num = len(self.values)
        self.reward_num = len(self.rewards)
        self.asset_map_nlayers = config['asset_map_nlayers']
        self.value_map_nlayers = config['value_map_nlayers']
        self.action_map_nlayers = config['action_map_nlayers']
        self.reward_map_nlayers = config['reward_map_nlayers']

        # Value Embeddings
        self.value_embeds = nn.Sequential(
            nn.Embedding(self.value_num, self.d_model),
            nn.Dropout(self.dropout))

        # Action Embeddings
        self.action_embeds = nn.Sequential(
            nn.Embedding(self.action_num, self.d_model),
            nn.Dropout(self.dropout))

        # Reward Embeddings
        self.reward_embeds = nn.Sequential(
            nn.Embedding(self.reward_num, self.d_model),
            nn.Dropout(self.dropout))

        # Time Encoding
        self.time_embeds = nn.Sequential(
            nn.Embedding(self.max_len, self.d_model),
            nn.Dropout(self.dropout))

        # Asset Data Encoder
        self.asset_embeds = Mapping(self.factor_num, self.d_model,
                                    self.asset_map_nlayers,
                                    'first', self.slope, self.dropout,
                                    True)

        # Decision Transformer
        self.dt = TransformerEnc(self.d_model, self.nhead, self.nlayers,
                                self.d_model * 2, self.dropout,
                                self.activation, True)

        # Value Prediction
        self.value_preds = Mapping(self.d_model, self.value_num,
                                self.action_map_nlayers,
                                'last', self.slope, self.dropout,
                                False)

        # Action Prediction
        self.action_preds = Mapping(self.d_model, self.action_num,
                                self.action_map_nlayers,
                                'last', self.slope, self.dropout,
                                False)

        # Reward Prediction
        self.reward_preds = Mapping(self.d_model, self.reward_num,
                                self.action_map_nlayers,
                                'last', self.slope, self.dropout,
                                False)

        # Expert Classifier
        self.expert_preds = torch.exp(self.values * self.K)
        self.expert_preds /= self.expert_preds.sum()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        " Initialize Model Weights "
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def device(self):
        return next(self.parameters()).device

    def calc_values(self, value_logits, sampling=False):
        """
            calculate value by expert_preds(inference time)

            Args:
                value_logits: value logit predictions by dt
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, value_num)
            Returns:
                value_preds: value prediction index
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
        """
        batch_size, seq_len, value_num = value_logits.shape

        with torch.no_grad():
            value_logits = value_logits.softmax(dim=-1).view(
                batch_size*seq_len, value_num)

            experts = self.expert_preds.view(1, value_num).to(self.device)

            value_preds = value_logits * experts
            value_preds =\
                value_preds / value_preds.sum(-1, keepdims=True)

            if sampling:
                value_preds = torch.multinomial(value_preds, 1)
            else:
                value_preds = value_preds.argmax(-1)

            value_preds = value_preds.view(batch_size, seq_len)

        return value_preds

    def calc_preds(self, logits, sampling=False):
        """
            calculate action and rewards(inference time)

            Args:
                logits: prediction logits by dt
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, dim)
            Returns:
                preds: prediction index
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
        """
        batch_size, seq_len, dim = logits.shape

        with torch.no_grad():
            logits = logits.softmax(dim=-1).view(
                batch_size*seq_len, dim)

            if sampling:
                preds = torch.multinomial(logits, 1)
            else:
                preds = logits.argmax(-1)

            preds = preds.view(batch_size, seq_len)

        return preds

    def forward(self, obs_in, values_in=None,
                actions_in=None, rewards_in=None,
                seq_mask=True, src_key_padding_mask=None):
        """
            Inference

            Args:
                obs_in: observations for Asset Factors
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, action_num, factor_num)
                values_in: value index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
                actions_in: action index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
                rewards_in: reward index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
            Returns:
                value_preds
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, value_num)
                action_preds
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, action_num)
                reward_preds
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, reward_num)
                outputs
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, action_num+3, seq_len, d_model)
        """
        batch_size, seq_len, action_num, _ = obs_in.shape

        seq_rng = torch.arange(seq_len).to(self.device)

        obs_cls = self.asset_embeds(obs_in)
        obs_cls = obs_cls.transpose(1, 2).contiguous()

        assert values_in is not None
        assert actions_in is not None
        assert rewards_in is not None
        assert values_in.shape[0] == batch_size
        assert actions_in.shape[0] == batch_size
        assert rewards_in.shape[0] == batch_size

        time_embeds = self.time_embeds(seq_rng).view(1, 1, seq_len, -1)
        time_embeds = time_embeds.repeat(batch_size, 1, 1, 1)

        value_embeds = self.value_embeds(values_in).unsqueeze(1)
        action_embeds = self.action_embeds(actions_in).unsqueeze(1)
        reward_embeds = self.reward_embeds(rewards_in).unsqueeze(1)

        obs = obs_cls + time_embeds
        values = value_embeds + time_embeds
        actions = action_embeds + time_embeds
        rewards = reward_embeds + time_embeds

        inputs = torch.cat(
            (obs, values, actions, rewards),
            dim=1).permute(0, 2, 1, 3).contiguous().view(
                batch_size, (action_num+3)*seq_len, -1)

        outputs = self.dt(inputs, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)
        outputs = outputs.view(
            batch_size, seq_len, (action_num+3), -1).permute(
                0, 2, 1, 3).contiguous()

        value_preds = self.value_preds(outputs[:, action_num-1])
        action_preds = self.action_preds(outputs[:, action_num])
        reward_preds = self.reward_preds(outputs[:, action_num+1])

        return value_preds, action_preds, reward_preds, outputs

    def inference(self, obs_in, values_in=None,
                actions_in=None, rewards_in=None, sampling=False,
                seq_mask=True, src_key_padding_mask=None):
        """
            inference

            Args:
                obs_in: observations for CrossAssetBert
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len, action_num, factor_num)
                values_in: value index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len-1)
                actions_in: action index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len-1)
                rewards_in: reward index series
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len-1)
        """
        self.eval()
        batch_size, seq_len, action_num, _ = obs_in.shape

        seq_rng = torch.arange(seq_len).to(self.device)
        time_embeds = self.time_embeds(seq_rng).view(1, 1, seq_len, -1)
        time_embeds = time_embeds.repeat(batch_size, 1, 1, 1)

        with torch.no_grad():
            if values_in is not None:
                value_cnt = values_in.shape[1]
                assert seq_len == (value_cnt + 1)

                value_embeds = self.value_embeds(values_in).unsqueeze(1)
            else:
                assert seq_len == 1
                value_cnt = None
                value_embeds = None

            if actions_in is not None:
                action_cnt = actions_in.shape[1]
                assert value_cnt == action_cnt

                action_embeds = self.action_embeds(
                    actions_in).unsqueeze(1)
            else:
                action_cnt = None
                action_embeds = None

            if rewards_in is not None:
                reward_cnt = rewards_in.shape[1]
                assert action_cnt == reward_cnt

                reward_embeds = self.reward_embeds(
                    rewards_in).unsqueeze(1)
            else:
                reward_cnt = None
                reward_embeds = None

            obs_cls = self.asset_embeds(obs_in)
            obs_cls = obs_cls.transpose(1, 2).contiguous()

            obs = obs_cls + time_embeds

            if value_cnt is not None:
                values = value_embeds + time_embeds[:, :, :value_cnt]
                actions = action_embeds + time_embeds[:, :, :value_cnt]
                rewards =\
                    reward_embeds + time_embeds[:, :, :value_cnt]

                inputs = torch.cat(
                    (obs[:, :, :-1], values, actions, rewards), dim=1)
                inputs = inputs.permute(0, 2, 1, 3).contiguous().view(
                    batch_size, (action_num+3)*value_cnt, -1)

                outputs = self.dt(inputs, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)
                outputs = outputs.view(
                    batch_size, value_cnt, (action_num+3), -1).permute(
                        0, 2, 1, 3).contiguous()

                value_logits = self.value_preds(
                    outputs[:, action_num-1, -1]).unsqueeze(-2)
                value_preds = self.calc_values(value_logits, sampling)

                value_new = self.value_embeds(value_preds)
                value_new = value_new + time_embeds[:, :, -1]

                inputs = torch.cat(
                    (inputs, obs[:, :, -1], value_new), dim=1)

                outputs = self.dt(inputs, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)

                action_logits = self.action_preds(
                    outputs[:, -1]).unsqueeze(1)
                action_preds = self.calc_preds(action_logits, sampling)
            else:
                obs = obs.squeeze(2)
                voutputs = self.dt(obs, seq_mask=seq_mask,
                            src_key_padding_mask=src_key_padding_mask)

                value_logits = self.value_preds(
                    voutputs[:, action_num-1]).unsqueeze(1)
                value_preds = self.calc_values(value_logits, sampling)

                value_embeds = self.value_embeds(value_preds)
                values = value_embeds + time_embeds[:, :, -1]

                inputs = torch.cat((obs, values), dim=1)

                outputs = self.dt(inputs, seq_mask=seq_mask,
                            src_key_padding_mask=src_key_padding_mask)

                action_logits = self.action_preds(
                    outputs[:, -1]).unsqueeze(1)
                action_preds = self.calc_preds(action_logits, sampling)

        return action_preds, value_preds


class BERTTA(nn.Module):
    """
        BERT for Technical Analysis
    """
    def __init__(self, config: dict = None):
        """
            Initialization

            Args:
                config: config file
                    * dtype: dict
        """
        super().__init__()

        if config is None:
            config = BERTTA_CONFIG
        self.config = config

        self.series_embeds_map_nlayers =\
            config['series_embeds_map_nlayers']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.d_model = config['d_model']
        self.dim_ff = config['dim_ff']
        self.nhead = config['nhead']
        self.nlayers = config['nlayers']
        self.activation = config['activation']
        self.max_len = config['max_len']
        self.task_num = config['task_num']
        self.task_map_nlayers = config['task_map_nlayers']
        self.recon_map_nlayers = config['recon_map_nlayers']

        # Positional Encoding
        # 0: CLS, 1~: Rest token(Reversed)
        self.positional_encoding = nn.Embedding(self.max_len+1,
                                            self.d_model)

        # CLS Token
        self.cls_token = nn.Embedding(1, self.d_model)

        # Series Embedding
        self.series_embeds = Mapping(1, self.d_model,
                                self.series_embeds_map_nlayers,
                                'first', self.slope, self.dropout,
                                True)

        # Transformer Encoder
        self.attn = TransformerEnc(self.d_model, self.nhead,
                            self.nlayers, self.dim_ff,
                            self.dropout, self.activation, True)

        # Task Predictor
        self.task_preds = nn.Sequential(
                        Mapping(self.d_model, self.task_num,
                                self.task_map_nlayers, 'last',
                                self.slope, self.dropout, False),
                        nn.Sigmoid())

        # Reconsruction
        self.recon_preds = Mapping(self.d_model, 1,
                                self.recon_map_nlayers,
                                'last', self.slope, self.dropout,
                                True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        " Initialize Model Weights "
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, series_in, mode='enc',
            seq_mask=False, src_key_padding_mask=None):
        """
            Inference

            Args:
                series_in: price series for an asset
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len)
                mode: inference mode of the model
                    * dtype: str
                    * kind of mode
                        * 'task': Task Prediction
                        * 'recon': Reconstruction
                        * 'enc': Encoding
                        * 'cls': CLS Token
                    * default: 'enc'
            Return:
                preds: prediction of each pretrain tasks
                    'task': (batch_size, task_num)
                    'recon': (batch_size, seq_len, 1)
                    'enc': (batch_size, seq_len+1, d_model)
                    'cls': (batch_size, d_model)
        """
        batch_size, seq_len = series_in.shape

        series_embeds = self.series_embeds(series_in.unsqueeze(-1))

        cls_token = self.cls_token(
            torch.tensor([[0]]).to(self.device))
        cls_token = cls_token.repeat(batch_size, 1, 1)
        series_embs = torch.cat((cls_token, series_embeds), dim=1)

        pe_cls = self.positional_encoding(
            torch.tensor([[0]]).to(self.device))
        pe_cls = pe_cls.repeat(batch_size, 1, 1)
        pe_rest = self.positional_encoding(
            torch.arange(
                seq_len-1, -1, -1).to(self.device)).view(1, seq_len, -1)
        pe_rest = pe_rest.repeat(batch_size, 1, 1)
        pe = torch.cat((pe_cls, pe_rest), dim=1)

        inputs = series_embs + pe
        embeds = self.attn(inputs, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)

        if mode == 'task':
            preds = self.task_preds(embeds[:, 0])
        elif mode == 'recon':
            preds = self.recon_preds(embeds[:, 1:])
        elif mode =='enc':
            preds = embeds
        elif mode == 'cls':
            preds = embeds[:, 0]

        return preds


class PortfolioRebalancer(nn.Module):
    """
        Portfolio Rebalancer Class
    """
    def __init__(self, config: dict = None):
        """
            Initialization

            Args:
                config: config file
                    * dtype: dict
        """
        super().__init__()

        if config is None:
            config = REBAL_CONFIG
        self.config = config

        self.factor_num = config['factor_num']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.nlayers = config['nlayers']
        self.activation = config['activation']
        self.stock_embeds_map_nlayers =\
            config['stock_embeds_map_nlayers']
        self.weights_nlayers = config['weights_nlayers']
        self.rebalancer_nlayers = config['rebalancer_nlayers']
        self.enc_path = config['enc_path']

        # Positional Encoding
        # 0: PE CLS
        # 1: PE Stock
        self.positional_encoding = nn.Embedding(2, self.d_model)

        # Segment Encoding
        # 0: SE CLS
        # 1: SE Recent
        # 2: SE Proposed
        self.segment_encoding = nn.Embedding(3, self.d_model)

        # CLS
        self.cls_token = nn.Embedding(1, self.d_model)
        self.weights_cls = nn.Embedding(1, self.d_model)

        # stock embedding
        self.stock_embeds = Mapping(self.factor_num, self.d_model,
                                    self.stock_embeds_map_nlayers,
                                    'first',self.slope,
                                    self.dropout, True)

        # Encoder: CrossAssetBERT
        self.enc = CrossAssetBERT(config=config)

        # Transformer Decoder
        self.dec = TransformerDec(self.d_model, self.nhead,
                                self.nlayers, self.d_model,
                                self.dropout, self.activation, True)

        # weights embedding
        self.weights_embeds = Mapping(1, self.d_model,
                                    self.weights_nlayers,
                                    'first', self.slope, self.dropout,
                                    True)

        # Rebalancer
        self.rebalancer = Mapping(self.d_model, 2,
                                self.rebalancer_nlayers,
                                'last', self.slope, self.dropout,
                                False)

        self.apply(self._init_weights)

        if self.enc_path is not None:
            self.enc.load_state_dict(
                torch.load(self.enc_path, map_location=self.device))

    def _init_weights(self, module):
        " Initialize Model Weights "
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, stocks_in=None,
            stocks_recent=None,
            stocks_proposed=None,
            weights_recent=None, weights_proposed=None,
            enc_time_mask=False, dec_time_mask=False,
            enc_key_padding_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None):
        """
            Inference

            Args:
                stocks_in: multifactor scores data for CrossAssetBert
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, in_stocks_num, factor_num)
                stocks_recent: multifactor scores data for recent portfolio
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, recent_stocks_num, factor_num)
                stocks_proposed: multifactor scores data for proposed portfolio
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, proposed_stocks_num, factor_num)
                weights_rec: recent portfolio weights
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, recent_stock_num)
                weights_proposed: proposed portfolio weights
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, proposed_stock_num)
            Return:
                rebal_preds: rebalancing selection
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, 2)
        """
        batch_size = stocks_in.shape[0]
        recent_stock_num = stocks_recent.shape[1]
        proposed_stock_num = stocks_proposed.shape[1]

        _, stocks_enc = self.enc(stocks_in, mode='enc',
                                seq_mask=enc_time_mask,
                                src_key_padding_mask=enc_key_padding_mask)

        # CLS
        pe_cls = self.positional_encoding(
            torch.tensor([0]).to(self.device)).view(1, 1, self.d_model)
        se_cls = self.segment_encoding(
            torch.tensor([0]).to(self.device)).view(1, 1, self.d_model)
        cls_t = self.cls_token(
            torch.tensor([0]).to(self.device)).view(1, 1, self.d_model)
        weights_cls = self.weights_cls(
            torch.tensor([0]).to(self.device)).view(1, 1, self.d_model)

        cls_token = cls_t + weights_cls + pe_cls + se_cls
        cls_token = cls_token.repeat(batch_size, 1, 1)

        # Portfolio Inputs
        pe_stock = self.positional_encoding(
            torch.tensor([1]).to(self.device)).view(1, 1, self.d_model)
        pe_stock = pe_stock.repeat(
            batch_size, recent_stock_num + proposed_stock_num, 1)

        se_rec = self.segment_encoding(
            torch.tensor([1]).to(self.device)).view(1, 1, self.d_model)
        se_rec = se_rec.repeat(batch_size, recent_stock_num, 1)
        se_prop = self.segment_encoding(
            torch.tensor([2]).to(self.device)).view(1, 1, self.d_model)
        se_prop = se_prop.repeat(batch_size, proposed_stock_num, 1)
        se_stock = torch.cat((se_rec, se_prop), dim=1)

        stocks_rec = self.stock_embeds(stocks_recent)
        stocks_prop = self.stock_embeds(stocks_proposed)

        weights_rec = self.weights_embeds(
            weights_recent.unsqueeze(-1))
        weights_prop = self.weights_embeds(
            weights_proposed.unsqueeze(-1))

        port_rec = stocks_rec + weights_rec
        port_prop = stocks_prop + weights_prop
        ports = torch.cat(
            (port_rec, port_prop), dim=1) + pe_stock + se_stock

        port_inputs = torch.cat((cls_token, ports), dim=1)

        decs = self.dec(port_inputs, stocks_enc,
                        tgt_mask=dec_time_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)

        rebal_preds = self.rebalancer(decs[:, 0])

        return rebal_preds


class InvestingPortfolioAllocator(nn.Module):
    """
        Investing Portfolio Allocator Class
    """
    def __init__(self, config: dict = None):
        """
            Initialization

            Args:
                config: config file
                    * dtype: dict
        """
        super().__init__()

        if config is None:
            config = IPA_CONFIG
        self.config = config

        self.factor_num = config['factor_num']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.nlayers = config['nlayers']
        self.activation = config['activation']
        self.port_type_num = config['port_type_num']
        self.stock_embeds_map_nlayers =\
            config['stock_embeds_map_nlayers']
        self.port_allocator_nlayers = config['port_allocator_nlayers']
        self.enc_path = config['enc_path']

        # Positional Encoding
        # 0: PE Port
        # 1: PE Stock
        self.positional_encoding = nn.Embedding(2, self.d_model)

        # Port Types Encoding
        self.port_types_embeds = nn.Embedding(
            self.port_type_num, self.d_model)

        # stock embedding
        self.stock_embeds = Mapping(self.factor_num, self.d_model,
                                    self.stock_embeds_map_nlayers,
                                    'first',self.slope,
                                    self.dropout, True)

        # Encoder: CrossAssetBERT
        self.enc = CrossAssetBERT(config=config)

        # Transformer Decoder
        self.dec = TransformerDec(self.d_model, self.nhead,
                                self.nlayers, self.d_model,
                                self.dropout, self.activation, True)

        # Portfolio Allocator
        self.port_allocator = Mapping(self.d_model, 1,
                                    self.port_allocator_nlayers,
                                    'last', self.slope, self.dropout,
                                    False)

        self.apply(self._init_weights)

        if self.enc_path is not None:
            self.enc.load_state_dict(
                torch.load(self.enc_path, map_location=self.device))

    def _init_weights(self, module):
        " Initialize Model Weights "
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, stocks_in=None,
            port_stocks=None,
            port_type_idx=None,
            enc_time_mask=False, dec_time_mask=False,
            enc_key_padding_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None):
        """
            Inference

            Args:
                stocks_in: multifactor scores data for observations
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, in_stocks_num, factor_num)
                port_stocks: multifactor scores data for rebalancing portfolios
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, port_stocks_now_num, factor_num)
                port_type_idx: portfolio strategy type index
                    * dtype: torch.LongTensor
                    * shape: (batch_size)
        """
        batch_size = stocks_in.shape[0]

        _, stocks_enc = self.enc(stocks_in, mode='enc',
                                seq_mask=enc_time_mask,
                                src_key_padding_mask=enc_key_padding_mask)

        port_types = self.port_types_embeds(
            port_type_idx).unsqueeze(1)

        # Meta informations
        pe_port = self.positional_encoding(
            torch.tensor([0]).to(self.device)).view(1, 1, self.d_model)

        pe_port = pe_port.repeat(batch_size, 1, 1)

        metas = port_types + pe_port

        # Portfolio Inputs for Inference
        port_stocks_now = self.stock_embeds(port_stocks)

        pe_stock = self.positional_encoding(
            torch.tensor([1]).to(self.device)).view(1, 1, self.d_model)
        pe_stock = pe_stock.repeat(
            batch_size, port_stocks.shape[1], 1)

        port_now = port_stocks_now + pe_stock
        port_inputs = torch.cat((metas, port_now), dim=1)

        decs = self.dec(port_inputs, stocks_enc,
                        tgt_mask=dec_time_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)

        weights_preds = self.port_allocator(decs[:, 1:])
        weights_preds = weights_preds.squeeze(-1).softmax(dim=-1)

        return weights_preds


class FineTunedCrossAssetBERT(nn.Module):
    """
        Finetuned Cross-Asset BERT
    """
    def __init__(self, config: dict = None):
        """
            initialization

            Args:
                config: config dict
        """
        super().__init__()

        if config is None:
            config = FT_CROSS_ASSET_BERT_CONFIG

        self.config = config

        self.d_model = config['d_model']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.up_daily_map_nlayers = config['up_daily_map_nlayers']
        self.down_daily_map_nlayers = config['down_daily_map_nlayers']
        self.pretrained_model_path =\
            config['pretrained_model_path']

        self.ca_bert = CrossAssetBERT(config)

        self.up_daily_preds = Mapping(self.d_model, 1,
                                    self.up_daily_map_nlayers,
                                    'last', self.slope, self.dropout,
                                    False)

        self.down_daily_preds = Mapping(self.d_model, 1,
                                    self.down_daily_map_nlayers,
                                    'last', self.slope, self.dropout,
                                    False)

        self.apply(self._init_weights)

        if self.pretrained_model_path is not None:
            self.ca_bert.load_state_dict(
                torch.load(self.pretrained_model_path,
                        map_location=self.device))

    def _init_weights(self, module):
        " Initialize Model Weights "
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, stocks_in, mode='up_daily',
            seq_mask=False, src_key_padding_mask=None):
        """
            Inference

            Args:
                stocks_in: multifactor scores data of stocks
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, stock_num, factor_num)
                mode: inference mode of the model
                    * dtype: bool
                    * kind of mode
                        * up_daily: Up Targets Prediction
                        * down_daily: Down Targets Prediction
                    * default: 'up_daily'
            Return:
                preds: predictions of each pretrain tasks
                    'up_daily'
                        * shape: (batch_size, stock_num)
                    'down_daily'
                        * shape: (batch_size, stock_num)
        """
        _, embeds = self.ca_bert(stocks_in, mode='enc',
                                seq_mask=seq_mask,
                                src_key_padding_mask=src_key_padding_mask)

        if mode == 'up_daily':
            preds = self.up_daily_preds(embeds[:, 1:]).squeeze(-1)
            preds = preds.softmax(dim=-1)
        elif mode == 'down_daily':
            preds = self.down_daily_preds(embeds[:, 1:]).squeeze(-1)
            preds = preds.softmax(dim=-1)

        return preds


class CrossAssetBERT(nn.Module):
    """
        Cross Asset BERT
    """
    def __init__(self, config: dict = None):
        """
            Initialization

            Args:
                config: config dict
        """
        super().__init__()

        if config is None:
            config = CROSS_ASSET_BERT_CONFIG
        self.config = config

        self.factor_num = config['factor_num']
        self.stock_embeds_map_nlayers =\
            config['stock_embeds_map_nlayers']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.nlayers = config['nlayers']
        self.activation = config['activation']
        self.recon_map_nlayers = config['recon_map_nlayers']
        self.up_map_nlayers = config['up_map_nlayers']
        self.down_map_nlayers = config['down_map_nlayers']
        self.market_map_nlayers = config['market_map_nlayers']

        # Positional Encoding
        # 0: CLS, 1: Rest token
        self.positional_encoding = nn.Embedding(2, self.d_model)

        # CLS Token
        self.cls_token = nn.Embedding(1, self.d_model)

        # stock embedding
        self.stock_embeds = Mapping(self.factor_num, self.d_model,
                                        self.stock_embeds_map_nlayers,
                                        'first',self.slope,
                                        self.dropout, True)

        # Transformer Encoder
        self.attn = TransformerEnc(self.d_model, self.nhead,
                            self.nlayers, self.d_model,
                            self.dropout, self.activation, True)

        # Up Targets Predictor
        self.up_preds = Mapping(self.d_model, 1,
                    self.up_map_nlayers, 'last',
                    self.slope, self.dropout, False)

        # Down Targets Predictor
        self.down_preds = Mapping(self.d_model, 1,
                    self.down_map_nlayers, 'last',
                    self.slope, self.dropout, False)

        # Stock Information Reconstructor
        self.recon_net = Mapping(self.d_model, self.factor_num,
                                self.recon_map_nlayers, 'last',
                                self.slope, self.dropout, False)

        # Market Up/Down Predictor
        self.market_preds = Mapping(self.d_model, 2,
                            self.market_map_nlayers, 'last',
                            self.slope, self.dropout, False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        " Initialize Model Weights "
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, stocks_in, mode='enc',
                seq_mask=False, src_key_padding_mask=None):
        """
            Inference

            Args:
                stocks_in: multifactor scores data of stocks
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, stock_num, factor_num)
                stocks_b: multifactor scores data of stocks
                    from different date that of assets_in
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, stock_num_b, factor_num)
                    * default: None
                mode: inference mode of the model
                    * dtype: bool
                    * kind of mode
                        * up: Up Targets Prediction
                        * down: Down Targets Prediction
                        * recon: Stock Information Reconstruction
                        * market: Market Up/Down Prediction
                        * enc: Get last output of self.attn
                    * default: 'enc'
            Return:
                preds: predictions of each pretrain tasks
                    'up'
                        * shape: (batch_size, stock_num)
                    'down'
                        * shape: (batch_size, stock_num)
                    'recon'
                        * shape: (batch_size, stock_num, factor_num)
                    'market'
                        * shape: (batch_size, 2)
                    'enc': None
                stocks: encodings of last layer of transformer
                    * shape: (batch_size, stock_num + 1, d_model)
        """
        batch_size = stocks_in.shape[0]
        stock_num = stocks_in.shape[1] + 1

        stocks_init = self.stock_embeds(stocks_in)

        cls_token = self.cls_token(
            torch.tensor([[0]]).to(self.device))
        cls_token = cls_token.repeat(batch_size, 1, 1)
        stocks_embs = torch.cat((cls_token, stocks_init), dim=1)

        pe_cls = self.positional_encoding(
            torch.tensor([[0]]).to(self.device))
        pe_cls = pe_cls.repeat(batch_size, 1, 1)
        pe_rest = self.positional_encoding(
            torch.tensor([[1]]).to(self.device))
        pe_rest = pe_rest.repeat(batch_size, stock_num - 1, 1)

        pe = torch.cat((pe_cls, pe_rest), dim=1)

        stocks = stocks_embs + pe
        stocks = self.attn(stocks, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)

        if mode == 'up':
            preds = self.up_preds(stocks[:, 1:]).squeeze(-1)
            preds = preds.softmax(dim=-1)
        elif mode == 'down':
            preds = self.down_preds(stocks[:, 1:]).squeeze(-1)
            preds = preds.softmax(dim=-1)
        elif mode == 'recon':
            preds = self.recon_net(stocks[:, 1:])
        elif mode == 'market':
            preds = self.market_preds(stocks[:, 0])
        else:
            preds = None

        return preds, stocks


class GlenScotia(nn.Module):
    """
        Glen Scotia Class
    """
    def __init__(self, config: dict = None):
        """
            Initialization

            Args:
                config: config dict
        """
        if config is None:
            config = GLEN_SCOTIA_CONFIG

        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.factor_num = config['factor_num']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.max_len = config['max_len']
        self.nhead = config['nhead']
        self.nlayers = config['nlayers']
        self.activation = config['activation']
        self.strategy_num = config['strategy_num']
        self.gen_num_layers = config['gen_num_layers']

        # Trading BERT
        self.bert = GlenEnc(config)

        # Time Encoding
        self.time_encoding = nn.Embedding(self.max_len, self.d_model)

        # Transformer Decoder
        self.dec = TransformerDec(self.d_model, self.nhead,
                                self.nlayers, self.d_model,
                                self.dropout, self.activation,
                                True)

        # Best-Worst Encoding
        # 0: Best
        # 1: Worst
        self.bw_encoding = nn.Embedding(2, self.d_model)

        # Rebalancing Encoding
        # 0: Initial Investing
        # 1: Rebalancing
        self.rebal_encoding = nn.Embedding(2, self.d_model)

        self.strategy_embeds =\
            nn.Embedding(self.strategy_num, self.d_model)

        self.strategy_generator = Mapping(self.d_model, self.strategy_num,
                                        self.gen_num_layers, 'last',
                                        self.slope, self.dropout, False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        " Initialize Model Weights "
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, assets_in, st_in, bw_in, rebal_in,
            enc_time_mask=False, dec_time_mask=True,
            enc_key_padding_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None):
        """
            inference

            Args:
                assets_in: assets in
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num, seq_len, factor_num)
                st_in: strategies in
                    * dtype: torch.LongTensor
                    * shape: (batch_size, st_num)
                bw_in: best worst index
                    * dtype: torch.LongTensor
                    * shape: (batch_size)
                    * 0: best
                    * 1: worst
                rebal_in: rebalancing index
                    * dtype: torch.LongTensor
                    * shape: (batch_size)
                    * 0: initial investing
                    * 1: rebalancing
            Return:
                st_preds: strategy predictions
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, st_num + 1, strategy_num)
        """
        batch_size = assets_in.shape[0]

        assets = self.bert(assets_in, seq_mask=enc_time_mask,
                        src_key_padding_mask=enc_key_padding_mask)

        bw = self.bw_encoding(bw_in).view(batch_size, 1, self.d_model)
        rebal = self.rebal_encoding(
            rebal_in).view(batch_size, 1, self.d_model)
        st_embeds = self.strategy_embeds(st_in)

        embs = torch.cat((bw, rebal, st_embeds), dim=1)

        seq = embs.shape[1]

        time_embeds = self.time_encoding(
            torch.arange(seq).to(self.device)).view(1, seq, self.d_model)
        time_embeds = time_embeds.repeat(batch_size, 1, 1)

        encodings = embs + time_embeds

        encodings = self.dec(encodings, assets,
                            tgt_mask=dec_time_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)


        st_preds = self.strategy_generator(encodings[:, 1:])

        return st_preds


class GlenEnc(nn.Module):
    """
        Trading BERT(Transformer Encoder)
            * Multiple Asset Series are expected to be given
    """
    def __init__(self, config: dict = None):
        """
            Initialization

            Args:
                config: config dict
        """
        if config is None:
            config = TRADING_BERT_CONFIG

        super().__init__()
        self.config = config
        self.factor_num = config['factor_num']
        self.asset_embeds_map_nlayers =\
            config['asset_embeds_map_nlayers']
        self.slope = config['slope']
        self.dropout = config['dropout']
        self.max_len = config['max_len']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.nlayers = config['nlayers']
        self.activation = config['activation']

        # Time Encoding
        self.time_encoding = nn.Embedding(self.max_len, self.d_model)

        # Segment Encoding
        self.seg_encoding = nn.Embedding(self.max_len, self.d_model)

        # Asset Embedding
        self.asset_embeds = Mapping(self.factor_num, self.d_model,
                                    self.asset_embeds_map_nlayers,
                                    'first', self.slope, self.dropout,
                                    True)

        # Transformer Encoder
        self.attn = TransformerEnc(self.d_model, self.nhead,
                                    self.nlayers, self.d_model,
                                    self.dropout, self.activation,
                                    True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        " Initialize Model Weights "
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, assets_in, seq_mask=False,
            src_key_padding_mask=None):
        """
            Inference

            Args:
                stocks_in: feature data of multiple assets
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num, seq_len, factor_num)
            Return:
                assets: None
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num*seq_len, d_model)
        """
        batch_size = assets_in.shape[0]
        asset_num = assets_in.shape[1]
        seq_len = assets_in.shape[2]

        # Time Encodings
        time_assets = self.time_encoding(
            torch.arange(seq_len).to(self.device))
        time_assets = time_assets.view(1, 1, seq_len, self.d_model)
        time_assets = time_assets.repeat(batch_size, asset_num, 1, 1)

        # Segment Encoding
        seg_embeds = self.seg_encoding(
            torch.arange(asset_num).to(self.device))
        seg_embeds = seg_embeds.view(1, asset_num, 1, self.d_model)
        seg_embeds = seg_embeds.repeat(batch_size, 1, seq_len, 1)

        # Asset Embeddings
        assets_init = self.asset_embeds(assets_in)

        asset_embeds = assets_init + seg_embeds + time_assets
        asset_embeds =\
            asset_embeds.permute(0, 2, 1, 3).contiguous().view(
                batch_size, asset_num*seq_len, self.d_model)

        # Enter Transformer
        assets = self.attn(asset_embeds, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)

        return assets