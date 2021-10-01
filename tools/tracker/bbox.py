# Copyright 2021-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import numpy as np


def rect_area( bb ):
    (lef,top,rig,bot) = bb
    return (rig-lef) * (bot-top)


def inter_over_union(a, b):
    ''' intersection over union
        intersection(a,b) / union(a, b)
    '''
    inter = intersection_area(a, b)
    return inter / (rect_area(a) + rect_area(b) - inter)


def intersection_area(a, b):
    ''' area of intersection(a, b)
    '''
    return intersection_line(a[0:4:2], b[0:4:2]) * intersection_line(a[1:4:2], b[1:4:2])


def intersection_line( a, b ):
    (x,y) = b
    (a,b) = a
    return (np.minimum(b,y) - np.maximum(a,x)).clip(min=0)
