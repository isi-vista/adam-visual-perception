# Part of the code in this file is adapted from https://bitbucket.org/phil_dias/gaze-estimation

from adam_visual_perception.gaze_estimation_model import prepare_modelSingle
from adam_visual_perception.utility import *

import keras.backend as K
import scipy.io as sio
import numpy as np
import glob
import cv2
import os


class BodyGazeEstimator:
    """ A class for estimating gaze ray from body landmarks """

    def __init__(
        self,
        model_weights="models/trainedOnGazeFollow_weights.h5",
        openpose_dir="/openpose",
        openpose_bin="./build/examples/openpose/openpose.bin",
        tmp_dir="tmp",
    ):
        """
        Parameters
        ----------
        model_weights : str, optional
            Path to pre-trained gaze estimation model weights
        openpose_dir : str, optional
            Path to OpenPose repository location
        openpose_bin : str, optional
            Path for openpose deployment command
        tmp_dir : str, optional
            Path to directory for temporary json files
        """
        # Load the gaze estimator pre-trained model
        self.model = prepare_modelSingle("relu")
        self.model.load_weights(model_weights)

        # Configure OpenPose paths
        self.openpose_dir = openpose_dir
        self.openpose_bin = openpose_bin
        self.tmp_dir = tmp_dir

    def get_gaze_rays(self, filename, shape):
        """
        Get the gaze rays for the given video file
        """
        # Call OpenPose
        this_path = os.getcwd()
        tmp_path = os.path.join(this_path, self.tmp_dir)
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        out_json_dir = os.path.join(tmp_path, "json")

        video_path = os.path.abspath(filename)
        video_path = video_path.replace(" ", "\ ")
        out_json_dir = out_json_dir.replace(" ", "\ ")

        # Change dir path to where OpenPose was installed
        os.chdir(self.openpose_dir)

        cmd = "{} --video {} --write_json {} --number_people_max 1 --display 0 --render_pose 0".format(
            self.openpose_bin, video_path, out_json_dir
        )
        print(cmd)
        os.system(cmd)

        # Go back to the original directory
        os.chdir(this_path)
        out_jsondir = os.path.join(tmp_path, "json")

        # Procss OpenPose results

        file_basename = os.path.basename(filename).split(".")[0]
        json_files = sorted(
            [
                j
                for j in glob.glob(
                    os.path.join(out_json_dir, file_basename + "_*.json")
                )
            ]
        )

        gaze_angles = {}

        # Iterate through json files for all frames
        for frame_no, json_filename in enumerate(json_files):
            _, suffix_ = os.path.split(json_filename)
            suffix_ = suffix_.split("_")
            suffix_ = suffix_[0]

            # collected poses from OpenPose detections
            poses, conf = load_many_poses_from_json(json_filename)

            assert len(poses) == 1, "More than one person detected"

            data = []
            itP = 0
            try:
                # compute facial keypoints coordinates w.r.t. to head centroid
                features, confs, centr = compute_head_features(poses[itP], conf[itP])
                # if minimal amount of facial keypoints was detected
                if features is not None:
                    featMap = np.asarray(features)
                    confMap = np.asarray(confs)

                    featMap = np.reshape(featMap, (1, 10))
                    confMap = np.reshape(confMap, (1, 5))

                    centr = np.asarray(centr)
                    centr = np.reshape(centr, (1, 2))

                    poseFeats = np.concatenate((centr, featMap, confMap), axis=1)

                    data.append(poseFeats)
            except Exception as e:
                print(e)

            # if at least one valid pose (person) was detected
            if data:
                # adjust features to feed our gaze NN
                ld = np.array(data)
                ld = np.squeeze(ld, axis=1)

                X_ = np.expand_dims(ld[:, 2:], axis=2)

                # deploy pre-trained gaze NN on these features
                # - prediction output: (X,Y,log of uncertainty)
                # -- (X,Y) are coordinates relative to centroid
                pred_ = self.model.predict(X_, batch_size=32, verbose=0)

                # Calculate the gaze ray
                centr = ld[itP, 0:2]
                res = pred_[itP, :-1]

                res[0] *= shape[0]
                res[1] *= shape[1]

                norm1 = res / np.linalg.norm(res)
                norm1[0] *= shape[0] * 0.15
                norm1[1] *= shape[0] * 0.15

                point = centr + norm1

                p1 = (int(centr[0]), int(centr[1]))
                p2 = (int(point[0]), int(point[1]))

                gaze_angles[frame_no] = (p1, p2)

        return gaze_angles
