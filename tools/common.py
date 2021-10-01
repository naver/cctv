# Copyright 2021-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os
import numpy as np
import torch


def mkdir_for( file_path ):
    os.makedirs(os.path.split(file_path)[0], exist_ok=True)
    return file_path


def model_size(model):
    ''' Computes the number of parameters of the model 
    '''
    size = 0
    for weights in model.state_dict().values():
        size += np.prod(weights.shape)
    return size


def select_device( gpu_idx ):
    """ set gpu_idx = -1 for CPU only, otherwise gpu_idx >= 0 represents the GPU index.
    """
    gpus = [gpu_idx]
    cuda = any(gpu>=0 for gpu in gpus)
    if cuda:
        assert all(gpu>=0 for gpu in gpus), 'cannot mix CPU and GPU devices'
        
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'],os.environ['CUDA_VISIBLE_DEVICES'])

        torch.backends.cudnn.benchmark = False # speed-up cudnn for constant shapes
        torch.backends.cudnn.fastest = True # even more speed-up?

        print( 'Launching on GPUs #' + os.environ['CUDA_VISIBLE_DEVICES'] )
    else:
        print( 'Launching on CPU only' )

    return torch.device('cuda' if cuda else 'cpu')


def todevice( obj, device ):
    """ Transfer an object to another device (i.e. GPU, CPU:torch, CPU:numpy).
    
    obj: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    """
    if isinstance(obj, dict):
        return {k:todevice(v, device) for k,v in obj.items()}
    
    if isinstance(obj, (tuple,list)):
        return type(obj)(todevice(x, device) for x in obj)

    if device == 'numpy':
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
    elif obj is not None:
        if isinstance(obj, np.ndarray):
            obj = torch.from_numpy(obj)
        obj = obj.to(device)
    return obj
