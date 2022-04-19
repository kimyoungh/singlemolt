"""
    Module for stock Transformers

    @author: Younghyun Kim
    Created on 2022.04.10
"""
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

        # Time Encoding
        self.time_encoding = nn.Embedding(self.max_len, self.d_model)

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

        # Asset Embeddings
        assets_init = self.asset_embeds(assets_in)

        asset_embeds = assets_init + time_assets
        asset_embeds =\
            asset_embeds.permute(0, 2, 1, 3).contiguous().view(
                batch_size, asset_num*seq_len, self.d_model)

        # Enter Transformer
        assets = self.attn(asset_embeds, seq_mask=seq_mask,
                        src_key_padding_mask=src_key_padding_mask)

        return assets