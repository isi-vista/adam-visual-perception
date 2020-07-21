""" Label Objects

This script, given a .tsv file containing video files, opens them one after
another, prompts the user to draw a bounding box around the four objects of
interest. Then, it asks to choose the object that is being gazed at by the
person in the video. The bounding boxes will target object index be written
in the same .tsv file. If the given .tsv file already has some of the inputs
labelled, the user will be prompted to only label non-labelled inputs

Parameters
----------
tsv_path : str, optional
    Path to tsv file containing dataset information (default is
    "benchmarks/gaze.tsv")
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
    tsv_path = "benchmarks/gaze.tsv"

@ex.automain
def main(_config):

    args = namedtuple('GenericDict', _config.keys())(**_config)

    # Setting the random seed
    np.random.seed(args.seed)

    # Load tsv
    if not os.path.isfile(args.tsv_path):
        raise Exception("The path to tsv file cannot be found at {}.".format(args.tsv_path))

    df = pd.read_csv(args.tsv_path, sep='\t')
    df = df.where(pd.notnull(df), None)

    obj1 = []
    obj2 = []
    obj3 = []
    obj4 = []
    labels = []
    for index, row in df.iterrows():
        row = row.tolist()
        l = len(row)

        # Filename, obj1, obj2, obj3, obj4, label
        if l == 6 and None not in row:
            filename, o1, o2, o3, o4, label = row
            obj1.append(o1)
            obj2.append(o2)
            obj3.append(o3)
            obj4.append(o4)
            labels.append(label)
        else:
            filename = row[0]
            obj1.append(draw_bbox(filename, "Select object 1"))
            obj2.append(draw_bbox(filename, "Select object 2"))
            obj3.append(draw_bbox(filename, "Select object 3"))
            obj4.append(draw_bbox(filename, "Select object 4"))
            label = input("Enter a the target object number: ")
            labels.append(label)

    # Save the bounding boxes in the dataframe
    df['obj1'] = obj1
    df['obj2'] = obj2
    df['obj3'] = obj3
    df['obj4'] = obj4
    df['label'] = labels

    convert_dict = {'label': int}
    df = df.astype(convert_dict)

    # Override the original tsv file
    df.to_csv(args.tsv_path, sep='\t', index=False)
