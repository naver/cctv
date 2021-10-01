# Copyright 2021-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

from pdb import set_trace as bb
import torch
from tools import common
from .transformer import Transformer


def load_net( model_path ):
    print(f"Loading network from {model_path} ...")
    model = torch.load(model_path, map_location=torch.device('cpu'))

    arch = f"{model['arch'][:-1]}, embdim=64, outdim=8)"
    print(f">> Building network = {arch}")
    net = eval(arch)
    net.load_state_dict(model['weights'])
    print(f"   model has {common.model_size(net)/10**6:.1f}M parameters")

    return net
