""" Preprocess videos

This script loads the videos, removes the audio track, cuts the videos from
the start point to endpoint for given dataset fragments, and saves them in a
new directory. A new `tsv` file is being generated with corresponding
filenames.

Parameters
----------
data_dir : str, optional
    Directory path to raw videos (default is "/path/to/raw/videos")
tsv_path : str, optional
    Path to tsv file containing dataset information (default is "benchmarks/gaze_raw.tsv")
target_dir : str, optional
    Where to save the preprocessed videos (default is "data/videos_gaze")
target_tsv : str, optional
    Where to write information on preprocessed vidoes (default is "benchmarks/gaze.tsv")
base_name : str, optional
    Base name of preprocessed vidoes (default is "video")
audio : bool, optional
    Whether to include the audio in the video files (default is False)
"""
# External imports
from sacred import Experiment
from sacred.observers import FileStorageObserver
from collections import namedtuple
from adam_visual_perception import preprocess
import pandas as pd
import numpy as np
import os

ex = Experiment()
#ex.observers.append(FileStorageObserver.create('sacred'))

@ex.config
def my_config():
    data_dir = "/path/to/raw/videos"
    tsv_path = "benchmarks/gaze_raw.tsv"
    target_dir = "data/videos_gaze"
    target_tsv = "benchmarks/gaze.tsv"
    base_name = "video"
    audio = False

@ex.automain
def main(_config):

    args = namedtuple('GenericDict', _config.keys())(**_config)

    # Setting the random seed
    np.random.seed(args.seed)

    # Check data dir
    if not os.path.isdir(args.data_dir):
        raise Exception("Data dir {} is invalid.".format(args.data_dir))

    # Load tsv
    if not os.path.isfile(args.tsv_path):
        raise Exception("The path to tsv file cannot be found at {}.".format(args.tsv_path))

    df = pd.read_csv(args.tsv_path, sep='\t')

    # Create the target dir
    if not os.path.isdir(args.target_dir):
        os.makedirs(args.target_dir)

    # Go through all video files, clip them and save as new files
    df_new = pd.DataFrame(columns = ['path'], index=range(0, len(df)))

    for index, row in df.iterrows():
        # Get the necessary information
        filename, start, end = row
        filename = os.path.join(args.data_dir, filename)
        target_name = os.path.join(args.target_dir, args.base_name + str(index) + ".mp4")

        # Preprocessing
        preprocess(filename, start, end, target_name, args.audio)

        # Save the entry in the target tsv file
        df_new.loc[index] = [target_name]

    # Dump the tsv file post-preprocessing
    df_new.to_csv(args.target_tsv, sep='\t', index=False)
    print("Dumped the tsv file of preprocessed videos at {}.".format(args.target_tsv))
