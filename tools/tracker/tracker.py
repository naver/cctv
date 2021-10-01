# Copyright 2021-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use
# Parts of the code in this file are from https://github.com/scanner-research/scanner (under Apache-2.0 license)

from collections import defaultdict
from tqdm import tqdm
import numpy as np


class Tracker:
    def __init__(self, kf_params, **kwargs):
        a, b1, b2, c, d, e, f = kf_params
        kf_params = dict(
            R_diag=[a, a, b1, b2],
            P_diag=[c, c, c, c, d, d, d],
            Q_diag=[e, e, e, e, f, f, f * f])

        raise NotImplementedError("TODO: insert your tracker init code here")
        self._my_object_tracker = None

    def push(self, boxes, scores):
        # non maxima suppression, remove overlapping boxes with lower scores
        idx_maxima = get_maxima_idx(boxes, scores)
        boxes = boxes[idx_maxima]
        scores = scores[idx_maxima]

        # tracker input: array Nx5 for N detections
        # where each row corresponds to a bounding box: [left, top, right, bottom, score]

        detection_tracker_in = np.c_[boxes, scores]
        result = self._my_object_tracker.update( detection_tracker_in )

        # Tracker returns a Mx6 array of M ongoing tracks
        # each row = [left, top, right, bottom, unique_track_id, input_box_idx]

        boxes = result[:, 0:4]
        uids = result[:, 4:5].astype(int)
        idxs = result[:, 5:6].astype(int)
        if len(result): idxs = idx_maxima[idxs] # mapping back to original box indexes before nms
        return uids, boxes, idxs


class TrackerSet:
    def __init__(self, categories, **kwargs):
        self._trackers = {category: Tracker(**kwargs) for category in categories}

    def push(self, boxes, scores, labels):
        tracked_boxes = []
        tracked_uids = []
        tracked_idxs = []
        for cls, tracker in self._trackers.items():
            sel = (labels == cls).nonzero()[0]
            track_uids, track_boxes, track_idxs = tracker.push(boxes[sel], scores[sel])
            
            tracked_idxs.append(sel[track_idxs])
            tracked_boxes.append(track_boxes)
            tracked_uids.append(track_uids)

        return np.vstack(tracked_boxes), np.vstack(tracked_uids), np.vstack(tracked_idxs)


def track_vehicles( dets, categories=None, update_box=True, dbg=(), **kwargs):
    from tools.tracks import enumerate_frames
    if not categories:
        categories = set(np.unique(dets['labels']).tolist())
    print(f'nb categories = {len(categories)}')

    tracker = TrackerSet(categories, **kwargs)
    tracks = defaultdict(list)
    dont_copy = {'img_hashes'}
    if update_box: dont_copy.add('boxes')

    for timestamp, sl in tqdm(enumerate_frames(dets), total=np.unique(dets['timestamps']).size):
        boxes  = dets['boxes'] [sl]
        scores = dets['scores'][sl]
        labels = dets['labels'][sl]
        new_boxes, uids, idxs = tracker.push(boxes, scores, labels)
        idxs = idxs.ravel()

        tracks['track_ids'].append( uids.ravel() )
        if update_box:
            tracks['boxes'].append( new_boxes.astype(np.float32) )
        for key in dets:
            if key in dont_copy: continue
            tracks[key].append( dets[key][sl][idxs] )

    # repack list of chunks 
    return {key:np.concatenate(vals, axis=0) for key, vals in tracks.items()}


def get_maxima_idx(boxes, scores, thr_iou=0.5, thr_score=0.3):
    """ Remove overlapping bounding boxes that have inferior scores.
    thr_iou: threshold minimum intersection over union
    thr_score: minimum score for a bounding box
    Returns a list of indices of the selected frames.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick_idx = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        idx = idxs[last]
        if scores[idx] < thr_score: break
        pick_idx.append(idx)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[idx], x1[idxs[:last]])
        yy1 = np.maximum(y1[idx], y1[idxs[:last]])
        xx2 = np.minimum(x2[idx], x2[idxs[:last]])
        yy2 = np.minimum(y2[idx], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap = intersection / original area
        overlap = (w * h) / areas[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > thr_iou)[0])))

    return np.int32(pick_idx)
