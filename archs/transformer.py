# Copyright 2021-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

from pdb import set_trace as bb

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer (nn.Module):
    """ Input shape = (seq_len, batch_size, inchan)
       Output shape = (seq_len, batch_size, outchan)
    """
    def __init__(self, embdim, outdim, nlayers=2, nhead=2, nhid=256, dropout=0.0):
        super().__init__()
        trf_layers = nn.TransformerEncoderLayer(embdim, nhead, nhid, dropout)
        self.transformer = nn.TransformerEncoder(trf_layers, nlayers)
        self.decoder = nn.Linear(embdim, outdim)
        self.src_mask = None
        self.embdim = embdim

    def forward(self, src, **kw):
        self.src_mask = None
        src = zero_pad(src, self.embdim)

        output = self.transformer(src.transpose(0,1), self.src_mask)
        output = self.decoder(output)
        return output.transpose(0,1)


def zero_pad( vec, embdim ):
    D = vec.shape[-1]
    assert D <= embdim
    res = torch.zeros(vec.shape[:-1]+(embdim,), dtype=torch.float32, device=vec.device)
    res[..., :D] = vec
    return res
