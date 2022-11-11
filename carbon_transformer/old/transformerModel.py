import numpy as np
import torch
import torch.nn as nn
from scipy.signal import sawtooth


class PositionalEncodingLayer(nn.Module):

    def __init__(self, attn_size, sample=128):
        super(PositionalEncodingLayer, self).__init__()
        pe = torch.zeros(sample, attn_size)
        position = torch.arange(0, sample, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, attn_size, 2).float() * (-9.21034037197618273607 / attn_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return x + self.pe



def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerModel(nn.Module):
    def __init__(self, ff_size=256, attn_size=128, heads_num=8, attn_depth=1, dropout=0.1, samp_len=128):
        super(TransformerModel, self).__init__()

        self.triang_mask = None
        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncodingLayer(attn_size=attn_size, sample=samp_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attn_size, nhead=heads_num, dropout=dropout)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=attn_size, nhead=heads_num, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=attn_depth)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=attn_depth)
        self.BN = nn.BatchNorm1d(samp_len)
        self.preDecoder = nn.Linear(ff_size, attn_size)
        self.postDecoder = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):
        self.preDecoder.bias.data.zero_()
        self.postDecoder.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                return nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, inp_data):
        if self.triang_mask is None or self.triang_mask.size(0) != len(inp_data):
            device = inp_data.device
            mask = _generate_square_subsequent_mask(len(inp_data)).to(device)
            self.triang_mask = mask

        output = self.pos_encoder(inp_data)

        output = self.transformer_encoder(output, self.triang_mask)
        output = self.BN(output)
        output = self.preDecoder(output)
        output = self.transformer_decoder(output, self.triang_mask)
        output = self.BN(output)
        output = self.postDecoder(output)

        return output
