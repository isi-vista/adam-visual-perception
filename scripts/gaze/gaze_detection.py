""" Cause Detection

This script runs the binary classifier that receives the video file and the bounding box of the target object as input and outputs whether it was moved by itself or due to an external force.

Parameters
----------
tsv_path : str, optional
    Path to tsv file containing dataset information (default is "benchmarks/gaze.tsv")
tracker_type : str, optional
    Tracking algorithm (default is "BOOSTING", possible values are ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"])
write_video : bool, optional
    Whether to save the processed video via OpenCV (default is True)
use_head : bool, optional
    Whether to use the head-landmark-based gaze detection method in the heuristic (default is True)
use_body : bool, optional
    Whether to use the body-landmark-based gaze detection method in the heuristic (default is True)
mix_priority : string, optional
    Which landmark-based gaze ray detection method to prioritise at every
    frame. Only relevant if previous two arguments are set to True. Possible
    values are "head", "body", "equal". The "equal" option makes use of both
    predicted gaze ray (default is "equal")
"""
# External imports
from sacred import Experiment
from sacred.observers import FileStorageObserver
from collections import namedtuple
from adam_visual_perception import GazePredictor
import pandas as pd
import numpy as np
import sys
import os

ex = Experiment()
ex.observers.append(FileStorageObserver.create('results'))

@ex.config
def my_config():
    tsv_path = "benchmarks/gaze.tsv"
    tracker_type = "CSRT"
    write_video = False
    use_head = True
    use_body = True
    mix_priority = "equal" # ["head", "body", "equal"]

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
    predictor = GazePredictor(
        tracker_type=args.tracker_type,
        write_video=args.write_video,
        use_head=args.use_head,
        use_body=args.use_body,
        mix_priority=args.mix_priority,
    )

    correct = 0
    total = 0

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

            pred = predictor.predict(filename, bboxes)
            if pred == label:
                correct += 1
            else:
                print("Incorrect prediction for {}. Predicted {} instead of {}.".format(filename, pred, label))
            total += 1
        else:
            print("Error: Did you forget to run the object labeling script?")
            sys.exit()

    print("Done!")
    print("Accuracy = {:.3f}%, {} out of {}.".format(correct * 100 / total, correct, total))
