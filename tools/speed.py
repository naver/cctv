# Copyright 2021-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

from pdb import set_trace as bb
import numpy as np
import torch

from tools.common import todevice, mkdir_for
from tools.tracks import enumerate_tracks, sort_tracks, filter_tracks, print_stats
from tools.geometry import normed, recover_homography_from_derivatives, jacobianh, applyh
numpy = lambda x: todevice(x, 'numpy')


def compute( dataset, model ):
    for video in dataset:
        print(f'>> Processing video {video}')
        estimate_homography( video, model )


def estimate_homography( video, net, motion_delta=0.04, **track_filters ):
    try:
        return video.homography

    except IOError as error:
        tracks = filter_tracks(video, video.tracks, sample_tracks=100, sample_boxes=10, **track_filters)

        tracks['motion'] = tracks['centers'].copy()
        for tid, track in enumerate_tracks(tracks, dic=True):
            track['motion'][:] = compute_motion(track['timestamps'], track['centers'], delta=motion_delta*video.fps)

        H_from_px = homography_from_transformer( video, tracks, net )

        np.savez( mkdir_for(error.filename), H_from_px=H_from_px)
        return H_from_px


def homography_from_transformer( video, tracks, net,  
        sampling_mode = 'random_tracks', sampling_size = 10, sampling_iters = 0,
        ransac_iters = 1024 ):

    sampled_tracks = sample_tracks( tracks, sampling_mode, sampling_size, sampling_iters)
    
    S = max(video.imsize)
    all_preds = []
    for sel in sampled_tracks:
        # extract a random subset of boxes
        boxes = {k:arr[sel] for k,arr in tracks.items()}
        assert len(boxes['timestamps'])

        # build embeddings
        embs = build_embeddings(boxes, S)

        # extract the homography
        embs = torch.from_numpy(embs)
        with torch.no_grad():
            preds = net(embs[None])
        all_preds.append( preds[0] )

    # ransac on all predictions
    H_from_px = ransac_homography_from_jacobians( torch.cat(all_preds), niter=ransac_iters)

    # scale back to image size
    H_from_px = H_from_px @ np.diag((1/S, 1/S, 1))
    return H_from_px


def sample_tracks( tracks, sampling_mode='random_tracks', sampling_size=10, sampling_iters=1 ):
    sampling_iters = sampling_iters or 99999
    
    if sampling_mode == 'random_boxes':
        nboxes = len(tracks['boxes'])
        sampling_iters = min(nboxes // sampling_size, sampling_iters)
        sel = np.random.choice(nboxes, size=(sampling_iters, sampling_size), replace=False)

    elif sampling_mode == 'random_tracks':
        track_ids = np.unique(tracks['track_ids'])
        ntracks = track_ids.size
        sampling_iters = min(ntracks, sampling_iters)
        sel_tracks = np.random.choice(track_ids, size=sampling_iters, replace=False)
        sel_tracks = [np.nonzero( tracks['track_ids'] == tid )[0] for tid in sel_tracks]
        sel = [np.random.choice(track, size=min(len(track), sampling_size), replace=False)
                for track in sel_tracks]
    else:
        raise ValueError(f'bad sampling mode: {sampling_mode}')
    return sel


def build_embeddings( boxes, scale, norm_motion=True ):
    centers, ellipsis = get_ellipsis(boxes['boxes'] / scale, boxes['masks'])
    motion = normed(boxes['motion'], axis=1)
    if norm_motion:
        motion *= np.sign(motion[:,1:2]) # make sure it's going down
    embs = np.c_[centers, 1001 * ellipsis, motion]
    return embs


def get_ellipsis(scaled_boxes, masks, mode=0.5):
    lt = scaled_boxes[:,0:2] # top-left corner
    wh = scaled_boxes[:,2:4] - lt # width, height of boxes

    # X,Y coordinate grid    
    H, W = masks.shape[-2:]
    xy = np.mgrid[0:H, 0:W].reshape(2, -1)[::-1].T
    # normalize xy in [0,1]
    if mode == 0:
        xy = xy / (W-1, H-1)
    if mode == 0.5:
        xy = (xy + 0.5) / (W, H)
    xy = xy[None] * wh[:,None] + lt[:,None]
    # corresponding weights
    w = masks.reshape(len(masks), -1)
    w = w / w.sum(axis=1, keepdims=True)
    assert np.isfinite(w).all(), bb()

    # weighted centers
    centers = (w[:, :, None] * xy).sum(axis=1, keepdims=True)
    xy -= centers

    # weithed covariance
    cov = w[:,None] * xy.transpose(0,2,1) @ xy
    return centers[:,0,:], cov.reshape(-1, 4)


def sqr_norm(x, **kw): 
    return np.square(x).sum(axis=-1)


def jacobian_from_preds(pred):
    # estimate jacobian from predicted output
    S, D = pred.shape
    pred = pred.view(S, 4, 2)
    centers = pred.mean(dim = -2)

    pred_jcam = torch.stack(((pred[...,2,:] - pred[...,1,:])/4, (pred[...,3,:] - pred[...,0,:])/2), dim=-2)
    pred_jw = torch.inverse( pred_jcam )
    return centers, pred_jw


def ransac_homography_from_jacobians(preds, niter=0):
    assert preds.ndim == 2
    pos, jw = numpy(jacobian_from_preds(preds))

    jcam = np.linalg.inv(jw)[...,::-1,:] # back to (dx,dy)
    jw = np.linalg.inv(jcam)

    jcam2 = jcam.copy()
    jcam2[..., 0, 1] = 0 # should be horizontal if there is no roll
    jw2 = np.linalg.inv(jcam2)
    jw2[:, 0, 1] = 0 # make sure it's zero
    norm_j = sqr_norm(jcam)

    # random order of triangular matrix
    N = len(jw)
    order = np.random.permutation((N-1)*(N-2)//2)

    best = 0, None
    for trial,o in enumerate(order[:niter or None]):
        # gets to indices from triangular number
        i = int(np.sqrt(8*o+1) + 1) // 2
        j = o - i*(i-1)//2

        # compute an hypothesis
        H_from_px = recover_homography_from_derivatives(pos[i], pos[j], jw2[i], jw2[j])

        # compute a robust fitting score
        jw_ = jacobianh(H_from_px, pos).T[:,:2]
        if np.isnan(jw_).any(): continue

        # dot-product normalized by largest norm
        # norm_dot_prod = |a| * |b| * cos(a,b) / (|a|*|b|) * min(|a|,|b|)/max(|a|,|b|)
        try: jcam_ = np.linalg.inv(jw_)
        except np.linalg.LinAlgError: continue # one of the matrix is singular
        score = (jcam * jcam_).sum(axis=2) / np.maximum(norm_j, sqr_norm(jcam_))
        score = score.prod(axis=1)

        score = np.sum( score )
        if score > best[0]: best = score, H_from_px

    assert best[1] is not None, bb()
    return best[1]


def compute_motion( timestamps, centers, delta=1, non_null=1.05 ):
    N, D = centers.shape
    assert len(timestamps) == N
    
    ts = timestamps
    c = centers
    # timestamps before / after
    delta /= 2
    tba = np.r_[ts - delta, ts + delta]
    tb_ta = tba.reshape(2,-1)

    for trial in range(999):
        # positions before / after
        before, after = np.c_[[np.interp(tba, ts, c[:,i]) for i in range(D)]].T.reshape(2,N,D)

        # increase the time interval until motion is non null
        nulls = (after == before).all(axis=1)
        if not(non_null and nulls.any()): break
        tb_ta[:,nulls] = ts[nulls] + non_null*(tb_ta[:,nulls] - ts[nulls])
    else:
        raise RuntimeError(f'could not get a non-null motion!\nts={ts}\ncenters={c}')

    # cropping timestamps to (ts[0], ts[-1])
    tb, ta = np.interp(tba, ts, ts).reshape(2,N,1)

    motion = (after - before) / (ta - tb)
    return motion


def compute_speed( track, H_from_px, video_fps, delta=5, dbg=()):
    centers_3d = applyh(H_from_px, track['centers'])

    ts = track['timestamps']
    speeds = np.linalg.norm(compute_motion(ts, centers_3d, delta=delta*video_fps), axis=1)
    speeds *= video_fps * 3.6 # meters/s to km/h
    
    assert np.isfinite(speeds).all(), bb()
    return speeds


def evaluate( dataset, 
        get_tracks = lambda video: sort_tracks(video.groundtruth_tracks), 
        gt_matcher = lambda video, tracks: (tracks['speeds'], np.abs(tracks['speeds']-tracks['est_speeds'])) ):

    all_diff_abs = []
    all_diff_rel = []

    for video in dataset:
        print(f'>> Evaluating video {video}')

        # load homography
        H_from_px = video.homography['H_from_px']

        # load tracks 
        tracks = get_tracks(video)
        print_stats(tracks)

        tracks['est_speeds'] = np.zeros_like(tracks['timestamps'], dtype='float32')
        for tid, track in enumerate_tracks(tracks, dic=True):
            track['est_speeds'][:] = compute_speed(track, H_from_px, video.fps)

        gt_speeds, diff = gt_matcher( video, tracks )
        print(f' >> Median error = {np.median(diff):.1f} km/h')
        all_diff_abs.append( diff )
        all_diff_rel.append( diff / gt_speeds )

    all_diff_abs = np.concatenate(all_diff_abs)
    all_diff_rel = 100 * np.concatenate(all_diff_rel)

    print("\nSummary:")
    print(f" >> absolute error: mean = {np.mean(all_diff_abs):.2f} km/h, median = {np.median(all_diff_abs):.2f} km/h")
    print(f" >> relative error: mean = {np.mean(all_diff_rel):.2f} %   , median = {np.median(all_diff_rel):.2f} %")
