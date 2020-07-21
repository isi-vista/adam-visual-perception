""" Cause Detection

This script runs the binary classifier that receives the video file and the bounding box of the target object as input and outputs whether it was moved by itself or due to an external force.

Parameters
----------
tsv_path : str, optional
    Path to tsv file containing dataset information (default is "benchmarks/motion.tsv")
tracker_type : str, optional
    Tracking algorithm (default is "BOOSTING", possible values are ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"])
thresh_conf : float, optional
    Confidence threshold for object detection of YOLO (default is 0.85)
use_gpu : bool, optional
    Whether to use a gpu (default is True)
write_video : bool, optional
    Whether to save the processed video via OpenCV (default is True)
yolo_path : str, optional
    Path to yolo weights and config files (default is "yolo")
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
ex.observers.append(FileStorageObserver.create('results'))

@ex.config
def my_config():
    tsv_path = "benchmarks/motion.tsv"
    tracker_type = "BOOSTING"
    thresh_conf = 0.85
    use_gpu = True
    yolo_path = 'yolo'
    write_video = False

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
        yolo_path=args.yolo_path,
        thresh_conf=args.thresh_conf,
        write_video=args.write_video,
    )

    correct = 0
    total = 0

    for index, row in df.iterrows():
        if len(row) == 3:
            filename, label, bbox = row
            if bbox is np.nan:
                print("Skipping {}. No bounding boxes are provided".format(filename))
            else:
                bbox = tuple(map(int, bbox.strip("()").split(', ')))
                pred = ot.predict(filename, bbox)
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
