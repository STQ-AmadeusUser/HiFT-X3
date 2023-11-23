import copy
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn import Dropout


class Cattention(nn.Module):
    def __init__(self, in_dim):
        super(Cattention, self).__init__()
        self.chanel_in = in_dim
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_dim*2, in_dim, kernel_size=1, stride=1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Conv2d(in_dim, in_dim // 6, 1, bias=False)
        self.linear2 = nn.Conv2d(in_dim // 6, in_dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x, y):
        ww = self.linear2(self.dropout(self.activation(self.linear1(self.avg_pool(y)))))  # 1x192x1x1
        weight = self.conv1(torch.cat((x, y), 1)) * ww  # 1x192x11x11
        return x + self.gamma * weight * x  # 1x192x11x11


class Transformer(Module):

    def __init__(self):
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer()
        self.encoder = TransformerEncoder(encoder_layer, 1)

        decoder_layer = TransformerDecoderLayer()
        self.decoder = TransformerDecoder(decoder_layer, 2)

        self.d_model = 192
        self.nhead = 6

    def forward(self, src: Tensor, srcc: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        # src: 121x1x192, srcc: 121x1x192, tgt: 121x1x192
        memory = self.encoder(src, srcc, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(192)

    def forward(self, src: Tensor,srcc: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(output, srcc, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(192)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):
    def __init__(self):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(192, 6, dropout=0.0)
        self.cross_attn = Cattention(192)

        # Implementation of Feedforward model
        self.eles = nn.Sequential(
                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, 192),
                nn.ReLU(inplace=True),
        )
        
        self.linear1 = nn.Linear(192, 2048)
        self.dropout = Dropout(0.0)
        self.linear2 = nn.Linear(2048, 192)
        self.norm0 = nn.LayerNorm(192)
        self.norm1 = nn.LayerNorm(192)
        self.norm2 = nn.LayerNorm(192)
        self.dropout1 = Dropout(0.0)
        self.dropout2 = Dropout(0.0)
        self.activation = F.relu

    def forward(self, src: Tensor, srcc: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        src2 = self.self_attn(self.norm0(src+srcc), self.norm0(src+srcc), src,
                              attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]  # 121x1x192

        src = src + self.dropout1(src2)  # 121x1x192
        src = self.norm1(src)  # 121x1x192

        src = self.cross_attn(
            src.permute(1, 2, 0).reshape(1, 192, 11, 11),  # 1x192x11x11
            srcc.permute(1, 2, 0).reshape(1, 192, 11, 11)  # 1x192x11x11
        ).reshape(1, 192, 121).permute(2, 0, 1)  # 121x1x192

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # 121x1x192
        src = src + self.dropout2(src2)  # 121x1x192
        src = self.norm2(src)  # 121x1x192

        return src  # 121x1x192


class TransformerDecoderLayer(Module):
    def __init__(self):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(192, 6, dropout=0.0)
        self.multihead_attn = MultiheadAttention(192, 6, dropout=0.0)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(192, 2048)
        self.dropout = Dropout(0.0)
        self.linear2 = nn.Linear(2048, 192)

        self.norm1 = nn.LayerNorm(192)
        self.norm2 = nn.LayerNorm(192)
        self.norm3 = nn.LayerNorm(192)
        self.dropout1 = Dropout(0.0)
        self.dropout2 = Dropout(0.0)
        self.dropout3 = Dropout(0.0)
        self.activation = F.relu

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]  # 121x1x192
        tgt = tgt + self.dropout1(tgt2)  # 121x1x192
        tgt = self.norm1(tgt)  # 121x1x192
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]  # 121x1x192
        tgt = tgt + self.dropout2(tgt2)  # 121x1x192
        tgt = self.norm2(tgt)  # 121x1x192
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))  # 121x1x192
        tgt = tgt + self.dropout3(tgt2)  # 121x1x192
        tgt = self.norm3(tgt)  # 121x1x192
        return tgt  # 121x1x192


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
