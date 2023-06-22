# Robust Automatic Monocular Vehicle Speed Estimation for Traffic Surveillance #
This repository contains the implementation of the following [paper](https://europe.naverlabs.com/research/publications/robust-and-automatic-monocular-vehicle-speed-estimation-for-traffic-surveillance/):

```text
@inproceedings{icctv,
  author    = {Jerome Revaud and Martin Humenberger},
  title     = {Robust Automatic Monocular Vehicle Speed Estimation for Traffic Surveillance},
  booktitle = {ICCV},
  year      = {2021},
}
```

License
-------
Our code is released under the Creative Commons BY-NC-SA 3.0 (see [LICENSE](LICENSE) for more details), available only for non-commercial use.


Requirements
------------
You need 
  - Python 3.8+ equipped with standard scientific packages and PyTorch / TorchVision:
    ```
    tqdm >= 4
    PIL >= 8.1.1
    numpy >= 1.19
    scipy >= 1.6.1
    torch >= 1.8.0
    torchvision >= 0.9.0
    cv2 >= 4.5.1
    filterpy >= 1.4.5
    ```
 - An object tracker. In the [ICCV paper](https://europe.naverlabs.com/research/publications/robust-and-automatic-monocular-vehicle-speed-estimation-for-traffic-surveillance/), 
   we used the [SORT](https://github.com/abewley/sort) tracker, but any object tracker of your preference can do.


Reproducing results on the BrnoCompSpeed dataset
------------------------------------------------

*Note*: Since we cannot share the 3D car models from the Unity library due to license issues, 
         this code only reproduces results from the learned method given a model pretrained based
         on these 3D car models.

1. Download BrnoCompSpeed dataset and evaluation code as explained in [JakubSochor's github](https://github.com/JakubSochor/BrnoCompSpeed).

2. Extract car tracks.

    `python run_brnocompspeed.py extract_tracks --dataset-dir /path/to/brnocompspeed --num-frames 5000`

    Here we are limiting the extraction to the first 5000 frames.
    It will save car detections (boxes) in `/path/to/brnocompspeed/detections`
    and tracks in `/path/to/brnocompspeed/tracks`.


3. Compute homographies given a pretrained model.

    `python run_brnocompspeed.py compute_homographies --dataset-dir /path/to/brnocompspeed --num-frames 5000 --model models/trained_transformer.pt`

    This will write all homographies in `/path/to/brnocompspeed/homography/`.

    *Note*: Since this process involves randomness due to RANSAC, you may obtain a slighlty different
    results compared to what is published in the paper. 
    Therefore we provide the homographies used in the paper in `data/brno_homographies.zip`.


4. Optionally, you can have a quick evaluation of the homographies.

    `python run_brnocompspeed.py evaluate_homographies --dataset-dir /path/to/brnocompspeed`

    *Note*: these results are computed based on **ground-truth boxes**, 
    hence the output results does not reflect the actual accuracy of the full system
    (i.e. detector, tracker and speed estimator jointly). Rather, it provides a quick
    estimates of how good are the estimated homographies based on ground-truth tracks.


5. Evaluate results using BrnoCompSpeed's evaluation code.

    ```bash
    # re-compute tracks, this time for entire videos (it will take a while)
    python run_brnocompspeed.py extract_tracks --dataset-dir /path/to/brnocompspeed 
    # export homographies and tracks in json format
    python run_brnocompspeed.py export_json --dataset-dir /path/to/brnocompspeed
    ```

    Then run the [evaluation code](https://github.com/JakubSochor/BrnoCompSpeed) (caution: it's written in python2).
    - First, set `RUN_FOR_SYSTEMS = ["transformer"]` in `/path/to/brnocompspeed/code/config.py`.
    - Then, execute the evaluation code:
      `cd /path/to/brnocompspeed/code && python eval.py -rc`


CCTV dataset
------------
We are currently working to release the CCTV dataset proposed in the paper.
In the meantime, please reach out to us if you need this dataset.
