"""
    Module for stock Transformers

    @author: Younghyun Kim
    Created on 2022.04.10
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from kingsman.cfg.knights_configs import TRADING_BERT_CONFIG
from kingsman.layers import TransformerEnc, TransformerDec, Mapping


class TradingBERT(nn.Module):
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
        self.recon_map_nlayers = config['recon_map_nlayers']
        self.comparing_map_nlayers = config['comparing_map_nlayers']
        self.overall_comparing_map_nlayers =\
            config['overall_comparing_map_nlayers']

        # Time Encoding
        # 0: CLS, 1~: Assets
        self.time_encoding = nn.Embedding(self.max_len, self.d_model)

        # CLS Token
        self.cls_token = nn.Embedding(1, self.d_model)

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

        # Reconstructor
        self.recon_net = Mapping(self.d_model, self.factor_num,
                                self.recon_map_nlayers, 'last',
                                self.slope, self.dropout, False)

        # Comparing Map
        self.comparing_net = Mapping(self.d_model, 1, 
                                    self.comparing_map_nlayers,
                                    'last', self.slope, self.dropout,
                                    False)

        # Overall Comparing Map
        self.overall_comparing_net =\
            Mapping(self.d_model, 2, self.overall_comparing_map_nlayers,
                    'last', self.slope, self.dropout, False)

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
                preds: predictions of each pretrain tasks
                    * dtype: TradingBERTPreds
                    'comps'
                        * shape: (batch_size, seq_len, asset_num)
                    'oc'
                        * shape: (batch_size, 2)
                    'recons'
                        * shape: (batch_size, asset_num, seq_len, factor_num)
                    'enc': None
                        * shape: (batch_size, (asset_num+1)*seq_len, d_model)
        """
        batch_size = assets_in.shape[0]
        asset_num = assets_in.shape[1]
        seq_len = assets_in.shape[2]

        # CLS Token
        cls_init = self.cls_token(
            torch.tensor([[0]]).to(self.device))
        cls_init = cls_init.repeat(batch_size, 1, 1)
        time_cls = self.time_encoding(
            torch.tensor([[0]]).to(self.device))
        time_cls = time_cls.repeat(batch_size, 1, 1)

        cls_token = cls_init + time_cls

        # Asset Embeddings
        assets_init = self.asset_embeds(assets_in)
        time_assets = self.time_encoding(
            torch.arange(1, seq_len+1).to(self.device))
        time_assets = time_assets.view(1, 1, seq_len, self.d_model)
        time_assets = time_assets.repeat(batch_size, asset_num, 1, 1)

        asset_embeds = assets_init + time_assets
        asset_embeds =\
            asset_embeds.permute(0, 2, 1, 3).contiguous().view(
                batch_size, asset_num*seq_len, self.d_model)

        assets = torch.cat((cls_token, asset_embeds), dim=1)

        # Enter Transformer
        assets = self.attn(assets, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)

        # comp
        comps = self.comparing_net(assets[:, 1:])
        comps = comps.view(batch_size, seq_len, -1)

        # oc
        oc = self.overall_comparing_net(assets[:, 0])

        # recon
        recons = self.recon_net(assets[:, 1:])
        recons = recons.view(batch_size, seq_len, asset_num, -1)
        recons = recons.permute(0, 2, 1, 3).contiguous()

        preds = TradingBERTPreds(
            enc=assets, comps=comps, oc=oc, recons=recons)

        return preds


@dataclass
class TradingBERTPreds:
    enc: torch.FloatTensor
    comps: torch.FloatTensor
    oc: torch.FloatTensor
    recons: torch.FloatTensor