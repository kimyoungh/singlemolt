"""
    Module for stock Transformers

    @author: Younghyun Kim
    Created on 2022.04.10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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