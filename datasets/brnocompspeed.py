# Copyright 2021-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

from pdb import set_trace as bb
from os.path import *
from collections import defaultdict
import pickle
from PIL import Image
import numpy as np

from .video_dataset import VideoDataset
from tools.geometry import *


class BrnoCompSpeed (VideoDataset):
    """ Annotation file contains:
     - fps: float
     
     - invalidLanes: set()
     - laneDivLines: list of 2d lines
     
     - distanceMeasurement: list of point pairs + distance
     - measurementLines: list of 2d lines
     
     - cars: list of annotated cars with their true speed and passage points
    """
    SESSIONS = [0,1,2,3,4,5,6]
    SIDES = ['left','center', 'right']
    COLLECTION = list(range(len(SESSIONS) * len(SIDES)))

    def __init__(self, video_num, root, **kw):
        super().__init__(video_num, 'video.avi', root=root, **kw)
        # load annotations
        with open(join(split(self.video_path)[0], "gt_data.pkl"), 'rb') as af:
            self.annots = pickle.load(af, encoding='latin-1')
        # override with the true values (video codec is broken?)
        true_fps = self.annots['fps'] / self.frame_step
        if self.__dict__.get('fps',0): self._video_nframes = int(self._video_nframes * true_fps // self.fps)
        self.fps = true_fps

    def _join_path(self, root, folder, *args):
        self.session = f"session{self.SESSIONS[self.video_num // 3]}_{self.SIDES[self.video_num % 3]}"
        if folder == 'videos':
            return join(root, "dataset", self.session, *args)
        else:
            ext = splitext(args[-1])[1]
            return join(root, folder, *args[:-1], self.session+ext )

    def __repr__(self):
        session = self.video_path.split('/')[-2]
        W,H = self.imsize
        return f"BrnoCompSpeed( #{self.video_num}={session}, {len(self)} frames, {W}x{H} pixels, {self.fps} fps )"

    @property
    def screen(self):
        return Image.open(self._join_path(self.root, 'videos', 'screen.png'))

    @property
    def video_mask(self):
        return np.array(Image.open(self._join_path(self.root, 'videos', 'video_mask.png'))) != 0

    @property
    def groundtruth_tracks(self):
        # prepare boxes in advance
        nLineIds = len(self.annots['measurementLines'])
        nLanes = len(self.annots['laneDivLines']) - 1
        rects = np.empty((nLineIds, nLanes, 4), dtype=np.int32)
        front = np.empty((nLineIds, nLanes, 2), dtype=np.int32)
        R = 0.6 * np.float32([[0,-1],[1, 0]])
        for lid,line in enumerate(self.annots['measurementLines']):
            pts = [line_intersection_2d(line, lane) for lane in self.annots['laneDivLines']]
            for i in range(len(pts)-1):
                p0, p1 = pts[i : i+2]
                front[lid,i] = (p0 + p1) / 2
                p2 = p0 + (p1 - p0) @ R
                assert p2[1] < max(p0[1], p1[1])
                p3 = p2 + p1 - p0
                pp = np.c_[p0,p1,p2,p3].T
                rects[lid,i] = np.r_[pp.min(0), pp.max(0)]

        boxes = []
        timestamps = []
        track_ids = []
        speeds = []
        centers = []
        for track_id, car in enumerate(self.annots['cars']):
            if not car['valid']: continue
            frames = car['intersections']
            lanes = list(car['laneIndex'])
            nd = len(frames)
            # assert nd == nLineIds, bb()
            # car_id = car['carId'] # sometimes not unique
            track_ids.append( np.full(nd, track_id, np.int32) )
            speeds.append( np.full(nd, car['speed'], np.float32) ) # in km/h
            
            for cp in car['intersections']:
                lid = cp['measurementLineId']
                frame = int(cp['videoTime'] * self.fps)
                timestamps.append( frame )
                boxes.append( rects[lid][lanes] )#.mean(axis=0).reshape(1,4) )
                centers.append( front[lid][lanes] )

        order = np.int32(timestamps).argsort()
        res = dict(timestamps=[timestamps], track_ids=track_ids, boxes=boxes, centers=centers, speeds=speeds)
        return {key:np.concatenate(vals)[order] for key,vals in res.items()}



if __name__ == '__main__':
    from .video_dataset import collection

    db = collection(BrnoCompSpeed, root='/local/cctv/BrnoCompSpeed/', frame_step=2, nframes=100)
    print(db)

    for video in db:
        print(video)
