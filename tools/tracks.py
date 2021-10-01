# Copyright 2021-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

from pdb import set_trace as bb
from copy import deepcopy
from tqdm import tqdm
import numpy as np

from tools.common import *


def extract( dataset ):
    """ Extract all vehicle tracks from a given video dataset, and save it to
        `output_dir/tracks/video_id/tracks.npz`
    """
    for video in dataset:
        print(f'>> Processing video {video} ...')
        track_cars(video)


def detect_cars(video, gpu_idx=0, batch_size=4, threads=8):
    try:
        return video.detections
    except IOError as error:
        # detect cars using MaskRCNN
        import torch

        print(f'\n>> Starting detection...')
        device = select_device(gpu_idx)
        gpu = lambda x: todevice(x,device)
        numpy = lambda x: todevice(x,'numpy')

        data_loader = torch.utils.data.DataLoader(video, batch_size=batch_size, shuffle=False)

        net = mask_rcnn() # load network
        net.eval()
        net.to(device)
            
        with torch.no_grad():
            dets = []
            for batch in tqdm(data_loader):
                frames = net(gpu(batch))
                for frame in frames: # convert masks to bytes
                    frame['masks'] *= 255.99 
                    frame['masks'] = frame['masks'][:,0,:,:].byte()
                dets += numpy(frames)

        print('>> Concatenating and saving...')
        frame_step = getattr(video, 'frame_step', 1)
        for idx, frame in enumerate(dets): 
            frame['timestamps'] = np.full_like(frame['labels'], idx)
        dets = {key:np.concatenate([frame[key] for frame in dets]) for key in dets[0]}

        np.savez_compressed( mkdir_for(error.filename), **dets)
        return dets    


def mask_rcnn(filter_classes=['car', 'motorcycle', 'truck', 'bus'], mask_subsample=2):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn, transform as trf
    from types import MethodType

    detector = maskrcnn_resnet50_fpn(pretrained=True)

    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    def dont_touch_masks(self, result, image_shapes, original_image_sizes):
        assert not self.training
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            result[i]["boxes"] = trf.resize_boxes(pred["boxes"], im_s, o_im_s)
            if "masks" in pred:
                masks = pred["masks"]
                if mask_subsample != 1:
                    masks = torch.nn.functional.avg_pool2d(masks, mask_subsample, stride=mask_subsample) if masks.numel() else masks[:,:,::mask_subsample,::mask_subsample] # empty
                result[i]["masks"] = masks
        return result

    # leave mask output as it is (do not post-process it to full image resize)
    detector.transform.postprocess = MethodType(dont_touch_masks, detector)

    if filter_classes:
        ok_classes = [COCO_CLASSES.index(c) for c in filter_classes]
        cls_map = torch.zeros(len(COCO_CLASSES), dtype=torch.int64)
        for i,c in enumerate(ok_classes, 1): cls_map[c] = i
        
        class FilteredDetector (type(detector)):
            CLASSES = {idx+1:name for idx,name in enumerate(filter_classes)}
            def __call__(self, *args, **kwargs):
                list_res = super().__call__(*args, **kwargs)
                for res in list_res:
                    valid = [l in ok_classes for l in res['labels']]
                    for key in res:
                        res[key] = res[key][valid]
                        if key == 'labels':
                            res[key] = cls_map[res[key]].to(res[key].device)
                return list_res
        # modify class type dynamically
        detector.__class__ = FilteredDetector
    return detector


def track_cars( video ):
    try:
        return video.tracks
    except IOError as error:
        from .tracker import tracker

        coef = np.sqrt(np.prod(video.imsize) / (480.*320.))
        assert coef > 0, 'image size is null'
        kf_params = [(coef * 5) ** 2,  # measure noise: variance of x,y
                     (coef * 5) ** 4,  # measure noise: variance of area
                     0.5 ** 2,  # measure noise: variance of aspect ratio (ar)
                     coef**2 * 100,  # init variance of x,y,area,ar
                     1e6,  # init variance of x,y,area speeds
                     coef ** 2 * 100,  # running variance of x,y,area,ar
                     coef**2 * 0.2]  # running variance of speeds

        detections = detect_cars(video)
        tracks = tracker.track_vehicles(detections, kf_params=kf_params, max_age=5, min_hits=1, update_box=False)

        np.savez_compressed( mkdir_for(error.filename), **tracks)
        return tracks


def enumerate_things( tracks, what, ids=None, dic=False, tqdm=False ):
    """ tracks: dict of arrays. Each array has a shape = (N, ...)
                where N is the total number of bounding boxes.

    tracks = {'timestamps': shape= (N,),
              'scores': shape = (N,),
              'labels': shape = (N,)
              'boxes': shape = (N, 4),
              'track_ids': shape = (N,),
               etc.}
    """
    # check if tracks are already sorted
    idxs = tracks[what]
    if not idxs.size: return
    well_sorted = len(idxs) == 1 or (idxs[1:] - idxs[:-1]).min() >= 0
    assert well_sorted, f"You need to sort_tracks(tracks, '{what}') beforehand"

    track_sizes = np.r_[0, np.bincount(idxs).cumsum()]
    if ids is None: ids = np.unique(idxs)
    if tqdm: ids = globals()['tqdm'](ids)
    for idx in ids:
        sl = slice(track_sizes[idx], track_sizes[idx+1])
        assert idx == idxs[sl.start] == idxs[sl.stop-1]
        yield (int(idx), {key:val[sl] for key,val in tracks.items()} if dic else sl)

def enumerate_frames( tracks, *args, **kw ):
    return enumerate_things(tracks, 'timestamps', *args, **kw)

def enumerate_tracks( tracks, *args, **kw ):
    return enumerate_things(tracks, 'track_ids', *args, **kw)


def sort_tracks( tracks, key='track_ids' ):
    idxs = tracks[key]
    if key != 'timestamps':
        # make sure that timestamps is the secondary order
        idxs = np.c_[tracks['timestamps'], idxs].astype(np.int32).view(np.int64).ravel()
    order = idxs.argsort()

    for k,vals in tracks.items():
        tracks[k] = vals[order] # modify input dictionary
    return tracks


def print_stats( tracks ):
    ts = tracks['timestamps']
    nfr = np.unique(ts).size
    if nfr == 0: print(">> empty tracks!"); return
    box_track_ids = tracks['track_ids']
    track_ids = np.unique(box_track_ids)
    
    from scipy.ndimage import minimum, maximum, sum
    nboxes_per_track = sum(np.ones_like(ts), labels=box_track_ids, index=track_ids) 
    track_len = ( maximum(ts, labels=box_track_ids, index=track_ids) 
                - minimum(ts, labels=box_track_ids, index=track_ids) )
    
    print(f">> found {track_ids.size} vehicle tracks from {nfr} frames and {len(ts)} boxes", 
          f"(track length = {np.median(nboxes_per_track):.1f}, duration = {np.median(track_len)})")

    from collections import namedtuple
    TrackStats = namedtuple('TrackStats', 'nframes ntracks nboxes nboxes_per_track')
    return TrackStats(nfr, track_ids.size, len(ts), nboxes_per_track)


def filter_tracks( video, tracks, rm_truck = 0, rm_boundary = 0.02, rm_masked = True,
                   rm_static = 0.5, sample_tracks = 0, sample_boxes = 0 ):
    tracks = deepcopy(tracks) # makesure we don't modify the original tracks

    if rm_truck>=0: tracks = remove_non_cars( tracks, min_num=rm_truck )
    if rm_boundary: tracks = remove_boundary_boxes( tracks, video.imsize, rm_boundary )
    if rm_masked:   tracks = remove_masked_boxes( tracks, video.video_mask )
    if rm_static:   tracks = remove_static_tracks( tracks, video.fps, min_len=5, iou_thr=rm_static )
    if sample_tracks: tracks = subsample_tracks( tracks, sample_tracks )

    tracks['boxes'] = clip_boxes( tracks['boxes'], video.imsize, min_car_size=5)
    tracks['centers'] = box_center( tracks['boxes'] )

    if sample_boxes: tracks = subsample_boxes( tracks, sample_boxes, prop_size=True )
    return tracks


def box_center( box ):
    return box.reshape(-1,2,2).mean(axis=1).squeeze()

def box_bottom_center( box ):
    return np.c_[box_center(box)[...,0], box[...,1::2].max(1)]

def box_wh( box ):
    return box[...,2:4] - box[...,0:2]

def box_area( box ):
    wh = box_wh( box )
    return np.prod(wh, axis=-1)

def ltrb_to_xywh( boxes, half=False ):
    xy = box_center( boxes )
    if half:
        return np.c_[xy, boxes[:,2:4]-xy]
    else:
        return np.c_[xy, boxes[:,2:4]-boxes[:,0:2]]

def valid_boxes(boxes, imsize, bnd=0.02):
    im_w, im_h = imsize
    l,t,r,b = boxes.T
    return np.c_[l > im_w*bnd, t > im_h*bnd, r < im_w*(1-bnd), b < im_h*(1-bnd)]

def clip_boxes(boxes, imsize, min_car_size=5):
    # we make sure that boxes are big enough (both width and height)
    # valid = Nx4 array, indicates which box coordinates are valid/invalid
    xywh = ltrb_to_xywh(boxes, half=True)
    too_small = (min_car_size/2 - xywh[:,2:4]).clip(min=0)

    if too_small.any():
        valid = valid_boxes( boxes, imsize).view(np.int8)
        xywh = xywh + np.c_[(valid[:,0:2] - valid[:,2:4]) * too_small, too_small]
    else:
        return boxes # unchanged input

    x,y,w,h = xywh.T
    return np.c_[x-w,y-h,x+w,y+h]


def remove_static_tracks( tracks, fps, min_len=5, iou_thr=0.5, time_gap=0.5):
    """ Remove trivially wrong tracks (based on simple tests).

    min_len: (int) minimum length of a track
    iou_thr: (float) maximum IoU that a track can have within itself.
    """
    from .tracker.bbox import inter_over_union

    # ts = tracks['timestamps']
    track_ids = tracks['track_ids']
    keep = np.zeros_like(track_ids)
    
    trks = [(track_ids == tid).nonzero()[0] for tid in np.unique(track_ids)][::-1]
    tid2 = 0
    while trks:
        sel = trks.pop()
    
        # not an empty track
        if len(sel) < min_len: continue

        # start and end are not overlapping
        boxes = tracks['boxes'][sel]
        start, end = boxes[[0,-1]]
        if inter_over_union(start, end) > 0: continue

        # same test for 30 km/h and 4m-long car
        # 10 km/h = 30/3.6 = 8.3 m/s
        # so 2 boxes must not overlap after 4/2.8 = 1.44s ~= 0.5s
        min_gap = int(np.ceil(fps * time_gap))

        # let's pretend that all timestamps are contiguous...
        overlaps = inter_over_union(boxes[min_gap:].T, boxes[:-min_gap].T)

        # cut into multiple sub-tracks each time it overlaps
        bad = (overlaps > iou_thr).nonzero()[0]
        if bad.size == 0: 
            tid2 += 1
            keep[sel] = tid2
            continue

        accu = np.zeros(len(boxes)+1, np.int32)
        accu[bad] += 1
        accu[bad+min_gap+1] -= 1
        good = (accu.cumsum()[:-1] == 0).view(np.int8) # all null segments are bad boxes
        good = np.r_[0, good, 0] # surrounded by bad segments
        start_end = good[1:] - good[:-1] # mark the beginning/end of all good segments
        for start, end in start_end.nonzero()[0].reshape(-1,2):
            # print(start, end)
            trks.append( sel[start:end] )

    # keep only valid instances
    valid = keep > 0
    return {key: val[valid] for key,val in tracks.items()}


def remap_labels( labels ):
    assert labels.min() >= 0
    remap = np.zeros(labels.max()+1, dtype=np.int32)
    tids = np.unique(labels)
    remap[tids] = np.arange(tids.size)
    return remap[labels]


def overlapping_tracks( tracks, targets, iou_thr=0.9, min_gap=5, min_num=1 ):
    """ This generator yields, for each target track, the mask of all similar tracks

    tracks: dictionary of tracks
    target: 1d array of track ids
    yields: array of boolean (one per box)
    """
    from tools.tracker.bbox import inter_over_union
    assert min_num >= 1 and iou_thr > 0 
    
    ts = tracks['timestamps']
    boxes = tracks['boxes']
    track_ids = tracks['track_ids']
    last_track = track_ids.max()

    # look for vehicle tracks that strongly overlap
    for tid in np.unique(targets):
        # timestamps of this track
        track = (track_ids == tid).nonzero()[0]
        trk_ts = ts[track] # track's timestamps
        ious = np.zeros((len(trk_ts), last_track+1), dtype=np.float32)

        for i, (t,sel) in enumerate(enumerate_frames(tracks, ids=trk_ts)):
            assert isinstance(sel, slice), "tracks must be ordered by timestamps"
            # compare this bounding box to all others
            box = boxes[track[i]] # this vehicle's box
            ious[i, track_ids[sel]] = inter_over_union( box, boxes[sel].T )

        # check which tracks are matching
        # ious = nframes(trk_ts) x ntracks
        # pl.plot(ious) # show all track temporal overlaps
        ious = (ious > iou_thr)
        t_start = trk_ts[ious.argmax(axis=0)]
        t_end = trk_ts[len(trk_ts)-1-ious[::-1].argmax(axis=0)]
        same_track = (ious.sum(axis=0) >= min_num) & (t_end - t_start >= min_gap)
        yield same_track


def remove_non_cars(tracks, car_label=1, min_num=0, iou_thr=0.9, min_gap=5 ):
    """ Remove tracks that are not cars, based on the labels.
    """
    sort_tracks(tracks, 'timestamps')
    try:
        labels = tracks['labels']
    except KeyError:
        return tracks # not labels, do nothing
    valid = np.isin(labels, car_label)

    if min_num > 0:
        track_ids = tracks['track_ids']
        track_ids[:] = remap_labels(track_ids) # renumber tracks for efficiency
    
        non_car_track_ids = np.unique(track_ids[~valid])
        for same_track in overlapping_tracks( tracks, non_car_track_ids, iou_thr, min_gap, min_num ):
            # remove tracks that are overlapping
            valid[same_track[track_ids]] = False
    
    return {key: val[valid] for key,val in tracks.items()}


def remove_boundary_boxes( tracks, imsize, frame_bnd=0.02):
    W, H = imsize
    boxes = tracks['boxes']
    valid = valid_boxes(boxes, imsize, bnd=frame_bnd).all(axis=1)
    return {key: val[valid] for key,val in tracks.items()}


def remove_masked_boxes( tracks, mask ):
    x, y = np.int32(0.5 + box_center(tracks['boxes'])).T
    valid = mask[y, x]
    return {key: val[valid] for key,val in tracks.items()}


def subsample_tracks( tracks, max_num, criterion='length'):
    tracks = sort_tracks(tracks)
    ts = tracks['timestamps']
    scores = {}
    for tid, track in enumerate_tracks(tracks):
        if criterion == 'length':
            scores[tid] = len(ts[track])
        else:
            raise ValueError(f'bad criterion {criterion}')

    sorted_scores = sorted(scores.items(), key=lambda p:p[1])
    whitelist = {tid for tid,_ in sorted_scores[-max_num:]}
    
    keep = []
    for tid, track in enumerate_tracks(tracks):
        if tid in whitelist:
            if isinstance(track, slice): track = np.mgrid[track]
            keep.append( track )
    keep = np.concatenate(keep)
    keep.sort() # keep original order
    return {key:val[keep] for key,val in tracks.items()}


def subsample_boxes( tracks, max_len, prop_size=True ):
    if prop_size:
        box_sizes = np.sqrt(box_area(tracks['boxes']))
    x = (np.arange(max_len) + 0.5 ) / max_len

    keep = []
    for tid, track in enumerate_tracks(tracks):
        if isinstance(track, slice): 
            track = np.mgrid[track]
        if len(track) <= max_len:
            keep.append( track )
            continue

        if prop_size:
            sizes = box_sizes[track]
            cum = np.r_[0, sizes.cumsum()]
        else:
            cum = np.arange(len(track)+1, dtype=np.float32)

        idxs = np.interp(x, cum / cum[-1], np.arange(len(track)+1))
        keep.append(track[np.int32(idxs)])

    keep = np.concatenate(keep)
    keep.sort() # keep original order
    return {key:val[keep] for key,val in tracks.items()}
