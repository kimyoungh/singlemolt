"""
    Strategy Generator Model based on Transformer

    @author: Younghyun Kim
    Created on 2022.03.18
"""
import torch
import torch.nn as nn

from kingsman.cfg.strategy_generator_configs import\
    SIMPLE_STRATEGY_GENERATOR_CONFIG
from kingsman.layers import Mapping, TransformerEnc, TransformerDec


class SimpleStrategyGenerator(nn.Module):
    """
        Simple Strategy Generator
    """
    def __init__(self, configs: dict = None):
        """
            Initialization
        """
        super().__init__()

        if configs is None:
            configs = SIMPLE_STRATEGY_GENERATOR_CONFIG

        self.configs = configs
        self.state_dim = configs['state_dim']
        self.map_nlayers = configs['map_nlayers']
        self.slope = configs['slope']
        self.dropout = configs['dropout']
        self.d_model = configs['d_model']
        self.enc_nheads = configs['enc_nheads']
        self.enc_nlayers = configs['enc_nlayers']
        self.enc_dim_ff = configs['enc_dim_ff']
        self.enc_activation = configs['enc_activation']
        self.dec_nheads = configs['dec_nheads']
        self.dec_nlayers = configs['dec_nlayers']
        self.dec_dim_ff = configs['dec_dim_ff']
        self.dec_activation = configs['dec_activation']
        self.out_map_nlayers = configs['out_map_nlayers']
        self.strategy_num = configs['strategy_num']
        self.max_len = configs['max_len']

        self.time_embeds = nn.Embedding(self.max_len, self.d_model)

        self.strategy_embeds = nn.Embedding(self.strategy_num,
                                        self.d_model)

        self.prefer_tokens = nn.Embedding(2, self.d_model)

        # rebal token
        #   0: initial investing
        #   1: rebalancing
        self.rebal_tokens = nn.Embedding(2, self.d_model)

        self.state_embeds = Mapping(self.state_dim, self.d_model,
                            self.map_nlayers, 'first', self.slope,
                            self.dropout, True)

        self.encoder = TransformerEnc(self.d_model, self.enc_nheads,
                                    self.enc_nlayers, self.enc_dim_ff,
                                    self.dropout, self.enc_activation,
                                    True)

        self.decoder = TransformerDec(self.d_model, self.dec_nheads,
                                    self.dec_nlayers, self.dec_dim_ff,
                                    self.dropout, self.dec_activation,
                                    True)

        self.strategy_generator =\
            Mapping(self.d_model, self.strategy_num,
                self.out_map_nlayers, 'last', self.slope,
                self.dropout, False)

        self.softmax = nn.Softmax(dim=-1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        " Initialize Model Weights"
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

    def get_state_encodings(self, states):
        """
            states: states
                * dtype: torch.FloatTensor
                * shape: (batch_size, asset_num, state_dim)
        """
        state_embs = self.state_embeds(states)
        state_embs = self.encoder(state_embs, seq_mask=False)

        return state_embs

    def forward(self, states, strategies,
                prefer_idx, rebal_idx, softmax=False):
        """
            Args:
                states: states
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, asset_num, state_dim)
                strategies: strategy sequence
                    * dtype: torch.LongTensor
                    * shape: (batch_size, seq_len)
                prefer_idx: preference index
                    * dtype: torch.LongTensor
                    * value
                        * 0: high return
                        * 1: low return
                    * shape: (batch_size)
                rebal_idx: rebalancing index
                    * dtype: torch.LongTensor
                    * value
                        * 0: initial investing
                        * 1: rebalancing
                    * shape: (batch_size)
                softmax: softmax to last head or not
                    * dtype: bool
            Return:
                st_preds: strategies predictions
                    * dtype: torch.FloatTensor
                    * shape: (batch_size, seq_len + 1, strategy_num)
        """
        seq_len = strategies.shape[1]

        state_embs = self.state_embeds(states)
        state_embs = self.encoder(state_embs, seq_mask=False)

        st_embeds = self.strategy_embeds(strategies)

        times = self.time_embeds(
            torch.arange(seq_len + 2).to(self.device))
        times = times.view(1, seq_len + 2, self.d_model)

        prefer_tokens = self.prefer_tokens(prefer_idx.unsqueeze(-1))
        rebal_tokens = self.rebal_tokens(rebal_idx.unsqueeze(-1))

        stacked_sts =\
            torch.cat(
                (prefer_tokens, rebal_tokens, st_embeds), dim=1) + times

        decodings = self.decoder(stacked_sts, state_embs, tgt_mask=True)

        strategy_preds =\
            self.strategy_generator(decodings[:, 1:])

        if softmax:
            strategy_preds = self.softmax(strategy_preds)

        return strategy_preds