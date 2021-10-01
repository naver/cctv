# Copyright 2021-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb; bb = pdb.set_trace
import numpy as np


def normed( vec, **kw ):
    return vec / np.linalg.norm(vec, keepdims=True, **kw)


def line_intersection_2d(line1, line2):
    ''' Intersection between 2 lines in 2d space.
        Each line is defined as (a,b,c) with a*x+b*y+c==0
    '''
    assert line1.shape == (3,) and line2.shape == (3,)
    res = np.cross(line1, line2)
    return res[:2] / res[2]


def pointToLineDistance(p, l):
    return abs(np.dot(l,p/p[2]))/np.linalg.norm(l[0:2])


def pointToLineProjection(l, p):
    p = p/p[-1]
    c = p[0]*l[1] - p[1]*l[0]
    perpendicularLine = np.array([-l[1], l[0], c])
    intersection = np.cross(l, perpendicularLine)
    return intersection/intersection[-1]


def isPointBetweenLines(p, l1, l2):
    return np.dot(p,l1)*np.dot(p,l2)*np.dot(l1[0:2],l2[0:2]) <= 0


def getLaneForPoint(p, lines):
    for i in range(len(lines)-1):
        if isPointBetweenLines(p, lines[i], lines[i+1]):
            return i
    return -1


def applyh(H, p, ncol=2, norm=True, front=False):
    """ Apply the homography to a list of 2d points in homogeneous coordinates.

    H: 3x3 matrix = Homography
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)
    
    ncol: int. number of columns of the result (2 or 3)
    norm: boolean. if True, the resut is projected on the z=1 plane.
    front: boolean or float. if not False, points that are behind the camera plane z=front are removed.
    
    Returns an array of projected 2d points.
    """
    if isinstance(H, np.ndarray):
        p = np.asarray(p)
    elif isinstance(H, torch.Tensor):
        p = torch.as_tensor(p, dtype=H.dtype)

    if p.shape[-1]+1 == H.shape[-1]:
        p = p @ H[:,:-1].T + H[:,-1]
    else:
        p = H @ p.T
        if p.ndim >= 2: p = p.swapaxes(-1,-2)
    if front is not False:
        p = p[p[...,-1] > front]
    if norm: 
        p = p / p[...,-1:]
    return p[...,:ncol]


def jacobianh(H, p):
    """ H is an homography that maps: f_H(x,y) --> (f_1, f_2)
    So the Jacobian J_H evaluated at p=(x,y) is a 2x2 matrix
    Output shape = (2, 2, N) = (f_, xy, N)

    Example of derivative:
                  numx    a*X + b*Y + c*Z
        since x = ----- = ---------------
                  denom   u*X + v*Y + w*Z

                numx' * denom - denom' * numx   a*denom - u*numx
        dx/dX = ----------------------------- = ----------------
                           denom**2                 denom**2
    """
    (a, b, c), (d, e, f), (u, v, w) = H
    numx, numy, denom = applyh(H, p, ncol=3, norm=False).T

    #                column x          column x
    J = np.float32(((a*denom - u*numx, b*denom - v*numx),  # row f_1
                    (d*denom - u*numy, e*denom - v*numy))) # row f_2
    return J / np.where(denom, denom*denom, np.nan)


def recover_homography_from_derivatives(p1, p2, j1, j2):
    """ p1, p2: 2 different points
        j1, j2: (transposed) jacobian at these points. 
                Normally j1[0,1] == j2[0,1] == 0 (moving on x in the image => no Y motion, i.e horizon is horizontal)

    Example:
        H = H_from_px # from pixels to meters
        j1, j2 = jacobianh(H, (p1,p2)).T[:,:2]
        H_ = recover_homography_from_derivatives(p1, p2, j1, j2)
        assert np.allclose(jacobianh(H_, (p1,p2)).T[:,:2], (j1,j2))
    """
    assert j1[0,1] == j2[0,1] == 0, "Horizon should be horizontal" # try calling upright_homography before
    x1,y1 = p1
    x2,y2 = p2
    K1 = j1[0,0]
    K2 = j2[0,0]
    v = (K2-K1) / (K1*y1 - K2*y2 + 1e-16)
    a = K2 + K2*y2*v
    e = j1[1,1] * (v*y1 + 1)**2
    b = j1[1,0] * (v*y1 + 1)**2 + x1 * a * v
    return np.float32((a,b,0,0,e,0,0,v,1)).reshape(3,3)
