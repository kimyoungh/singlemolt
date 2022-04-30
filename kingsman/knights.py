"""
    Module for stock Transformers

    @author: Younghyun Kim
    Created on 2022.04.10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from kingsman.cfg.knights_configs\
    import TRADING_BERT_CONFIG, GLEN_SCOTIA_CONFIG
from kingsman.layers import TransformerEnc, TransformerDec, Mapping


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
        self.bert = TradingBERT(config)

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