""" Label Objects

This script, given a .tsv file containing video files, opens them one after
another, prompts the user to draw a bounding box around the object that will
be serving as a target in that video. The bounding boxes will be written in
the same .tsv file. If the given .tsv file already has some of the inputs
labelled, the user will be prompted to only label non-labelled inputs.

Parameters
----------
tsv_path : str, optional
    Path to tsv file containing dataset information (default is
    "benchmarks/motion.tsv")
"""


# External imports
from sacred import Experiment
from sacred.observers import FileStorageObserver
from collections import namedtuple
from adam_visual_perception.utility import draw_bbox
import pandas as pd
import numpy as np
import os
import sys

ex = Experiment()
#ex.observers.append(FileStorageObserver.create('sacred'))

@ex.config
def my_config():
    tsv_path = "benchmarks/motion.tsv"

@ex.automain
def main(_config):

    args = namedtuple('GenericDict', _config.keys())(**_config)

    # Setting the random seed
    np.random.seed(args.seed)

    # Load tsv
    if not os.path.isfile(args.tsv_path):
        raise Exception("The path to tsv file cannot be found at {}.".format(args.tsv_path))

    df = pd.read_csv(args.tsv_path, sep='\t')
    #df = df.where(pd.notnull(df), None)

    bboxes = []
    for index, row in df.iterrows():
        if len(row) == 2:
            filename, label = row
            # Draw bounding boxes
            bboxes.append(draw_bbox(filename))
        elif len(row) == 3:
            filename, label, bbox = row
            if bbox is np.nan:
                bboxes.append(draw_bbox(filename))
            else:
                bboxes.append(bbox)
        else:
            print("Error: Row = {}".format(row))
            sys.exit()

    # Save the bounding boxes in the dataframe
    df['object'] = bboxes

    # Override the original tsv file
    df.to_csv(args.tsv_path, sep='\t', index=False)
