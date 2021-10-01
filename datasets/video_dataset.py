# Copyright 2021-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os
from os.path import *
import numpy as np

from .video import Video


class VideoDataset (Video):
    """ Each instance is a single video, but the collection can be accessed as:

    >>> for video in collection(MyVideoClass, **options):
    >>>    ...
    """
    VIDEO_FOLDER = 'videos'
    _join_path = lambda self, *args: join(*args)
    
    def __init__(self, video_num, video_name, root=None, **video_options):
        self.root = root
        self.video_num = video_num
        self.video_name = video_name
        video_path = self._join_path(root, self.VIDEO_FOLDER, video_name)
        Video.__init__(self, video_path, **video_options)
        self.set_video_options(**video_options)

    def set_video_options(self, **options):
        assert self.frame_step == options.get('frame_step',self.frame_step), 'not implemented'
        for k,v in options.items(): setattr(self,k,v)
        opt = '' 
        if self.frame_step != 1: opt += f'_framestep{self.frame_step}'
        if self.nframes:     opt += f'_nframes{self.nframes}'
        if self.start_frame: opt += f'_startframe{self.start_frame}'
        self.options_str = opt[1:]
        self._set_paths(self.video_name)

    def _set_paths(self, video_name):
        self.detections_path =  self._join_path(self.root, 'detections', self.options_str, video_name+'.npz')
        self.tracks_path =      self._join_path(self.root, 'tracks', self.options_str, video_name+'.npz')
        self.homography_path =  self._join_path(self.root, 'homography', video_name+'.npz')

    @property
    def detections(self):
        return self._load_dic(self.detections_path)

    @property
    def tracks(self):
        return self._load_dic(self.tracks_path)

    @property
    def homography(self):
        return self._load_dic(self.homography_path)

    def _load_dic(self, path):
        return dict(np.load(path))


class collection:
    """ iterate over the full collection 
    if dummy: but without actually loading videos
    """
    def __init__(self, cls, **options):
        self.cls = cls
        self.video_nums = cls.COLLECTION
        self.options = options

    def __repr__(self):
        return f"collection of {len(self)} {self.cls.__name__}({', '.join(f'{k}={v}' for k,v in self.options.items())})"

    def __len__(self):
        return len(self.video_nums)

    def __getitem__(self, idx):
        video = self.cls(self.video_nums[idx], **self.options)
        video.__cmd__ = f"{self.cls.__name__}({self.video_nums[idx]},{','.join(str(k)+'='+str(v) for k,v in self.options.items())})".replace(',)',')')
        return video
