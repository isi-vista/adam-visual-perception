""" Test Object Tracking

This script receives a .tsv file as input which has already been labelled
and runs the four selected objects tracking algorithm on all videos.
The target object that is being gazed at by the person is presented in blue.

Parameters
----------
tsv_path : str, optional
    Path to tsv file containing dataset information (default is "benchmarks/gaze.tsv")
tracker_type : str, optional
    Tracking algorithm (default is "CSRT", possible values are ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'])
use_gpu : bool, optional
    Whether to use a gpu (default is False)
write_video : bool, optional
    Whether to save the processed video via OpenCV (default is True)
"""
# External imports
from sacred import Experiment
from sacred.observers import FileStorageObserver
from collections import namedtuple
from adam_visual_perception import ObjectTracker
import pandas as pd
import numpy as np
import sys
import os

ex = Experiment()

@ex.config
def my_config():
    tsv_path = "benchmarks/gaze.tsv"
    tracker_type = "CSRT"
    use_gpu = False
    write_video = True

@ex.automain
def main(_config):

    args = namedtuple('GenericDict', _config.keys())(**_config)

    # Setting the random seed
    np.random.seed(args.seed)

    # Load tsv
    if not os.path.isfile(args.tsv_path):
        raise Exception("The path to tsv file cannot be found at {}.".format(args.tsv_path))

    df = pd.read_csv(args.tsv_path, sep='\t')

    # Definea tracker
    ot = ObjectTracker(
        tracker_type=args.tracker_type,
        use_gpu=args.use_gpu,
        detect_objects=False,
        write_video=args.write_video,
    )

    for index, row in df.iterrows():
        if len(row) == 6:
            filename, o1, o2, o3, o4, label = row
            print("Started {}".format(filename))
            objs = [o1, o2, o3, o4]
            bboxes = []
            for bbox in objs:
                if bbox is np.nan:
                    print("Skipping {}. No bounding boxes are provided".format(filename))
                else:
                    bbox = tuple(map(int, bbox.strip("()").split(', ')))
                    bboxes.append(bbox)

            ot.get_four_bboxes(filename, bboxes, label, args.write_video)
        else:
            print("Error: Did you forget to run the object labeling script?")
            sys.exit()

    print("Done!")
