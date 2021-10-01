# Copyright 2021-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os
import argparse
import numpy as np

from datasets import collection, BrnoCompSpeed
from tools import common, tracks, speed
from archs import load_net


parser = argparse.ArgumentParser('Run experiments on BrnoCompSpeed')

parser.add_argument('todo', choices=[
        'extract_tracks',
        'compute_homographies',
        'evaluate_homographies',
        'evaluate_speeds',
        'export_json'])
parser.add_argument('--dataset-dir', type=str, required=True, help='BrnoCompSpeed directory')
parser.add_argument('--num-frames', type=int, default=0, help='crop video to N frames')

parser.add_argument('--model-path', type=str, default=None, help='Path to model')

args = parser.parse_args()


# Create dataset object. Each element is a video.
dataset = collection(BrnoCompSpeed, root=args.dataset_dir, frame_step=2, nframes=args.num_frames)

if args.todo == 'extract_tracks':
    tracks.extract(dataset)

if args.todo == 'compute_homographies':
    assert args.model_path
    model = load_net(args.model_path)
    speed.compute(dataset, model)

if args.todo == 'evaluate_homographies':
    speed.evaluate( dataset )

if args.todo == 'export_json':
    import json
    name = 'transformer'

    def get_tracks( video ):
        from tools.tracks import sort_tracks, filter_tracks, box_bottom_center
        tracks = sort_tracks(filter_tracks(video, video.tracks))
        tracks['centers'] = box_bottom_center(tracks['boxes'])
        return tracks

    def tracks_to_json( tt, H_from_px, scale=1 ):
        res = {}
        # road geometry
        res["camera_calibration"] = dict(H_from_px = H_from_px.ravel().tolist(), scale=scale)

        # car detections
        res["cars"] = cars = []
        for cid, track in tracks.enumerate_tracks(tt, dic=True):
            cx, cy = track['centers'].T.tolist()
            cars.append(dict(id=cid, frames = track['timestamps'].tolist(), posX = cx, posY = cy))
        return res

    for video in dataset:
        json_path = os.path.join(args.dataset_dir, 'results', video.session, 'system_'+name+'.json')
        print(f'>> Exporting {json_path}')
        if os.path.isfile(json_path): raise IOError('File exists: '+json_path) 

        car_tracks = get_tracks(video)
        car_tracks['timestamps'] *= video.frame_step
        H_from_px = video.homography['H_from_px']

        data = tracks_to_json(car_tracks, H_from_px)
        with open(common.mkdir_for(json_path), 'w') as f:
            f.write(json.dumps(data))

    print(f"cd {args.dataset_dir}/code && python eval.py -rc")
