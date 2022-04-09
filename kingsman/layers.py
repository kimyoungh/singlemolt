"""
    jayoo fundamental NN Architectures

    @author: Younghyun Kim
    Created on 2022.02.07
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEnc(nn.Module):
    """
        Transformer Encoder
    """
    def __init__(self, d_model, nhead=4, nlayers=6,
                dim_feedforward=2048, dropout=0.1,
                activation='relu', batch_first=True):
        """
            batch_first: batch dimension(default: False)
                * True: input shape -> (batch_size, seq_len, emb_size)
                * False: input shape -> (seq_len, batch_size, emb_size)
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.batch_first = batch_first

        self.enc_layer =\
            nn.TransformerEncoderLayer(d_model, nhead,
                                    dim_feedforward,
                                    dropout, activation,
                                    batch_first=batch_first)

        self.attn_enc = nn.TransformerEncoder(self.enc_layer,
                                            num_layers=nlayers)

    @property
    def device(self):
        return next(self.parameters()).device

    def generate_square_subsequent_mask(self, seq_len):
        """
            generate Square Subsequent Mask
        """
        mask = (torch.triu(torch.ones((seq_len, seq_len),
        device=self.device)) == 1).transpose(0, 1)

        mask = mask.float().masked_fill(mask == 0,
            float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def forward(self, x_in, seq_mask=False, src_key_padding_mask=None):

        if self.batch_first:
            seq_len = x_in.shape[1]
        else:
            seq_len = x_in.shape[0]

        if seq_mask:
            mask = self.generate_square_subsequent_mask(seq_len)
        else:
            mask = None

        out_embs = self.attn_enc(x_in, mask=mask,
                            src_key_padding_mask=src_key_padding_mask)

        return out_embs


class TransformerDec(nn.Module):
    """
        Transformer Decoder
    """
    def __init__(self, d_model, nhead=4, nlayers=6,
                dim_feedforward=2048, dropout=0.1,
                activation='relu', batch_first=True):
        """
            batch_first: batch dimension(default: False)
                * True: input shape -> (batch_size, seq_len, emb_size)
                * False: input shape -> (seq_len, batch_size, emb_size)
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.batch_first = batch_first

        self.dec_layer =\
            nn.TransformerDecoderLayer(d_model, nhead,
                                    dim_feedforward, dropout,
                                    activation, batch_first=batch_first)

        self.attn_dec = nn.TransformerDecoder(self.dec_layer,
                                            num_layers=nlayers)

    @property
    def device(self):
        return next(self.parameters()).device

    def generate_square_subsequent_mask(self, seq_len):
        """
            generate Square Subsequent Mask
        """
        mask = (torch.triu(torch.ones((seq_len, seq_len),
        device=self.device)) == 1).transpose(0, 1)

        mask = mask.float().masked_fill(mask == 0,
            float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def forward(self, tgt, enc_memory,
                tgt_mask=False, memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        if self.batch_first:
            seq_len = tgt.shape[1]
        else:
            seq_len = tgt.shape[0]

        if tgt_mask:
            mask = self.generate_square_subsequent_mask(seq_len)
        else:
            mask = None

        out_embs =\
            self.attn_dec(tgt, enc_memory, tgt_mask=mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)

        return out_embs


class MLPMixer(nn.Module):
    """
        MLP Mixer
    """
    def __init__(self, input_dim, channel_dim,
                seq_len, nlayers=2,
                slope=0.2, dropout=0.1):
        """
            Initialization

            Input Shape: (batch_size X seq_len X input_dim)
        """
        super().__init__()

        self.input_dim = input_dim
        self.channel_dim = channel_dim
        self.seq_len = seq_len
        self.nlayers = nlayers
        self.slope = slope
        self.dropout = dropout

        self.in_net = nn.Sequential(
            nn.Linear(input_dim, channel_dim),
            nn.LeakyReLU(slope),
        )

        self.mixer_layers = nn.ModuleList()

        for i in range(nlayers):
            layer = MLPMixerLayer(channel_dim, seq_len,
                                slope, dropout)
            self.mixer_layers.append(layer)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x_in):
        """
            x_in: (batch_size X seq_len X input_dim)
            x_out: (batch_size X seq_len X channel_dim)
        """
        x_out = self.in_net(x_in)

        for i in range(self.nlayers):
            x_out = self.mixer_layers[i](x_out)

        return x_out


class MLPMixerLayer(nn.Module):
    """
        MLP Mixer Block
    """
    def __init__(self, channel_dim=16, seq_len=250,
                slope=0.2, dropout=0.1):
        """
            Initialization

            Input Shape: (batch_size X seq_len X channel_dim)
        """
        super().__init__()
        self.channel_dim = channel_dim
        self.seq_len = seq_len
        self.slope = slope
        self.dropout_p = dropout

        self.layer_norm_1 = nn.LayerNorm(channel_dim)
        self.mlp_1 = Mapping(seq_len, seq_len, 2, 'first',
                            slope, dropout, False)

        self.layer_norm_2 = nn.LayerNorm(channel_dim)
        self.mlp_2 = Mapping(channel_dim, channel_dim,
                            2, 'first', slope, dropout, False)

        self.dropout = nn.Dropout(dropout)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x_in):
        """
            x_in: (batch_size X seq_len X channel_dim)
        """
        x_1 = self.layer_norm_1(x_in).transpose(1, 2).contiguous()
        x_1 = self.mlp_1(x_1).transpose(1, 2).contiguous()

        x_2 = x_1 + self.dropout(x_in)
        x_3 = self.mlp_2(self.layer_norm_2(x_2))

        x_out = x_3 + self.dropout(x_2)

        return x_out


class Mapping(nn.Module):
    """
        Mapping Network
    """
    def __init__(self, in_dim, out_dim,
                num_layers=8, out_dim_pos='last',
                slope=0.2, dropout=0.1,
                last_activation=True):
        """
            Args:
                in_dim: input dim
                out_dim: output dim
                num_layers: # of layers
                out_dim_pos: out_dim을 적용할 layer 위치(default: last)
                    * first: 첫번째 layer out_dim
                    * last: 마지막 layer out_dim
                slope: negative slope for leaky relu
                dropout: dropout
                last_activation: Bool.(default: True)
                    * True: 마지막 layer에 leaky relu 적용
                    * False: 마지막 layer에 leaky relu 미적용
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.out_dim_pos = out_dim_pos
        self.slope = slope
        self.dropout = dropout
        self.last_activation = last_activation

        self.map_blocks = nn.ModuleList()

        in_d = in_dim

        for i in range(num_layers):
            if i < num_layers - 1:
                if out_dim_pos == 'last':
                    out_d = in_dim
                elif out_dim_pos == 'first':
                    out_d = out_dim
                block = MappingBlock(in_d, out_d, slope, dropout)
            else:
                out_d = out_dim
                if last_activation:
                    block = MappingBlock(in_d, out_d, slope, dropout)
                else:
                    block = nn.Linear(in_d, out_d)
                    nn.init.kaiming_normal_(block.weight)
                    if block.bias is not None:
                        with torch.no_grad():
                            block.bias.zero_()
            in_d = out_d
            self.map_blocks.append(block)

    def forward(self, x_in):
        " x_in forward "
        for i in range(self.num_layers):
            x_in = self.map_blocks[i](x_in)

        return x_in


class MappingBlock(nn.Module):
    " Default Linear Mapping Block "
    def __init__(self, in_dim, out_dim,
                leak_slope=0.2, dropout=0.1, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.leak_slope = leak_slope
        self.dropout = dropout
        self.bias = bias

        self.fc_net = nn.Linear(in_dim, out_dim, bias=bias)

        self.leaky_relu = nn.LeakyReLU(leak_slope)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_normal_(self.fc_net.weight)

        if self.fc_net.bias is not None:
            with torch.no_grad():
                self.fc_net.bias.zero_()

    def forward(self, x_in):
        " forward "
        x_out = self.leaky_relu(self.fc_net(x_in))
        x_out = self.dropout(x_out)

        return x_out