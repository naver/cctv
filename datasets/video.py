# Copyright 2021-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

from pdb import set_trace as bb
from os.path import join

import numpy as np
import cv2
from torchvision import transforms


class Video:
    """ Video object. Frames can ba accessed (almost) randomly:

    >>> video = Video(path)
    >>> frame_20 = video[20] --> torch.FloatTensor
    >>> frame_10 = video[10] --> torch.FloatTensor
    """
    def __init__(self, video_path, nframes=0, frame_step=1, start_frame=0, 
                       cache_size=256):
        assert isinstance(nframes, int)
        assert isinstance(start_frame, int)
        assert isinstance(frame_step, int)
        self.video_path = video_path
        self.frame_step = frame_step
        self.start_frame = start_frame
        self.nframes = nframes
        self.transform = transforms.ToTensor()

        self._init_video(cache_size)

    def _init_video(self, cache_size):
        self.video = cv2.VideoCapture(self.video_path)
        self._video_nframes = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS) / self.frame_step
        self._cache_size = cache_size or float('inf')
        self._cached_frames = {}
        self._cur_frame = 0

    def __repr__(self):
        return f"Video('{self.video_path}', {len(self)} frames)"

    def __len__(self):
        return max(0, min(self.nframes or 999999, 1 + (self._video_nframes - 1) // self.frame_step - self.start_frame))

    @property 
    def imsize(self): 
        # returns (width, height)
        return self.shape[1::-1]

    @property 
    def shape(self):  
        # returns (height, width, #channels)
        if not self._cached_frames: self[0] # access the first frame
        frame = next(iter(self._cached_frames.values())) 
        return frame.shape

    @property 
    def imcenter(self):
        return (np.float32(self.imsize) - 1) / 2

    def __getitem__(self, idx):
        if not(0 <= idx < (self.nframes or 999999)): 
            raise IndexError()
        idx += self.start_frame
        if idx * self.frame_step > self._video_nframes: 
            raise IndexError()
        if idx not in self._cached_frames:
            self._fill_cache( idx )
        frame = self._cached_frames[idx]
        return self.transform(frame)

    def _fill_cache(self, idx):
        if self._cached_frames and idx < next(iter(self._cached_frames)):
            raise ValueError(f'Cannot rewind video more than {self._cache_size} frames')
            # THIS IS BROKEN:
            #self.video.set(cv2.CAP_PROP_POS_FRAMES, idx-1) 
            #self._cur_frame = idx-1

        for _ in range(self._cur_frame*self.frame_step, self.start_frame*self.frame_step):
            flag, frame = self.video.read()
            if not flag: return # should not happen: start_frame > video_nframes
        self._cur_frame = max(self._cur_frame, self.start_frame)

        while self._cur_frame <= idx:
            for _ in range(self.frame_step):
                flag, frame = self.video.read()
                if not flag: # no more frame, returns the last one
                    frame = self._cached_frames[max(self._cached_frames)]
                    break
            # delete old frames so that cache size stays reasonable
            if len(self._cached_frames) >= self._cache_size:
                del self._cached_frames[ next(iter(self._cached_frames)) ]
            self._cached_frames[self._cur_frame] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._cur_frame += 1


if __name__ == '__main__':
    db = Video('/local/cctv/BrnoCompSpeed/dataset/session0_right/video.avi')
    print(db)
    
    from matplotlib import pyplot as pl
    for frame in range(100):
        pl.clf()
        pl.imshow(db[frame].permute(1,2,0))
        pl.pause(0.001)
