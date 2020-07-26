# Gaze Object Detection

### Abstract

In our goal to further understand the capability of computational systems to derive various features of input representation, we investigate whether they can extract information about where people are looking at any given time. Specifically, we aim to understand whether such systems can determine which objects in the room are being gazed at by the human.

### Task

Given a video segment and four objects in the video, determine which object is being gazed at by the human.

### Method

We conduct **object tracking** with respect to all four objects to understand where they are at every frame. We make use of real-time object tracking using the [OpenCV](https://opencv.org/) implementation of the [CSRT](https://openaccess.thecvf.com/content_cvpr_2017/html/Lukezic_Discriminative_Correlation_Filter_CVPR_2017_paper.html) tracker.

We then try to estimate where exactly the person is looking at during each frame. Specifically, we aim to derive the **gaze ray** of the person that is starting from her nose. To this end, we make use of two methods:

- **Head pose estimation**, i.e., understanding how the head of a person is tilted with respect to the camera. We utilise the [Dlib's facial landmark detector](http://dlib.net/face_landmark_detection.py.html) to identifying the 2D coordinates of facial landmarks (e.g. nose, chin, ears, eyes, etc). These coordinates, alongside 3D world coordinates, can be used to derive the approximate 3D coordinates of the landmarks, which, in turn, can be used to derive the gaze ray. We make use of the OpenCV's [solvePnP()](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html) method for 3D reconstruction.
- **Body pose estimation**, i.e., understanding how the entire body of a person is posed with respect to the camera. We use the popular [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) library to derive the coordinates of the body landmarks (e.g. nose, neck, shoulders, etc) and pass them to the pre-trained model by [Dias et al. (2019)](https://arxiv.org/pdf/1909.09225.pdf) that predicts the necessary gaze ray.

Our heuristic method is based on the abovementioned components. At each step, we predict the gaze ray using either (or both) estimators, track the four objects in the frame, and calculate the shortest distance between our gaze ray and midpoints of boundary boxes of corresponding objects. The object that is in the shortest distance to the ray is being chosen for the frame, while the mode of the per-frame predictions is being chosen for the entire video.

We gave our heuristic the ability to use both estimators at the same time (for each frame) by calculating the *average* gaze ray or prioritising one of them in case both estimators yield a result. 

### Results

The table below illustrates the results of our experiments on 50 video segments.

| Estimator | Priority | Accuracy |  Correct  |
|:-:|:-:|:-:|:-:|
| Head | |80% | 40 |
| Body | |96% | 48 |
| Head + Body | Head | 92% | 46 |
| Head + Body | Body | 96% | 48 |
| Head + Body | Equal | 98% | 49 |

The *mixed* heuristic that equally prioritises head- and body-landmark based gaze estimators outperform other configurations with 98% accuracy (49 out of 50).

To give an intuition on how our method works, we provide a [video](https://youtu.be/3gW4axWnF5E) that illustrated the gaze ray based on which one of the objects in the bounding boxes is chosen.

# Scripts: Step-by-step guide

Please follow the following steps to replicate the results of our experiments:

- [Data Preprocessing](#data-preprocessing): before running our methods on video clips, we preprocess them. Specifically, we cut the video segments of our interest, make the size smaller, etc,
- [Target Object Labeling](#target-object-labeling): once video clips are ready, we can run a script that will allow users to identify the four target objects and select the one that is being gazed at,
- [Testing Object Tracking](#testing-object-tracking): in case you want to take a quick look into how the tracking methods work for the labelled objects,
- [Gaze Object Detector](#gaze-object-detector): classifying the object that is being gazed at in the videos.

## Data Preprocessing

The following script loads the videos, removes the audio track, cuts the videos from the start point to endpoint for given dataset fragments, and saves them in a new directory. A new `tsv` file is being generated with corresponding filenames and ground truth labels.

```bash
python3 scripts/gaze/preprocess.py [with NAME_1=VAL_1 ... NAME_N=VAL_1]
```

where the optional arguments for the script are as follows:

 * data_dir - Directory path to raw videos (default is "/path/to/raw/videos")
 * tsv_path - Path to tsv file containing dataset information (default is "benchmarks/gaze_raw.tsv")
 * target_dir - Where to save the preprocessed videos (default is "data/videos_gaze")
 * target_tsv - Where to write information on preprocessed vidoes (default is "benchmarks/gaze.tsv")
 * base_name - Base name of preprocessed vidoes (default is "video")
 * audio - Whether to include the audio in the video files (default is False)

**Note**: You don't need to perform this step unless you have extended the benchmark data.

 ## Target Object Labeling

This script, given a .tsv file containing video files, opens them one after another, prompts the user to draw a bounding box around the four objects of interest. Then, it asks to choose the object that is being gazed at by the person in the video. The bounding boxes will target object index be written in the same .tsv file. If the given .tsv file already has some of the inputs labelled, the user will be prompted to only label non-labelled inputs

 ```bash
python3 scripts/gaze/label_objects.py [with tsv_path=VAL]
 ```

where the optional arguments for the script are as follows:

 * tsv_path - Path to tsv file containing dataset information (default is "benchmarks/gaze.tsv")

**Note**: You don't need to perform this step unless you have extended the benchmark data.

## Testing Object Tracking

The following script receives a .tsv file as input which has already been labelled and runs the four selected objects tracking algorithm on all videos. The target object that is being gazed at by the person is shown in blue.

 ```bash
python3 scripts/gaze/object_tracking.py [with NAME_1=VAL_1 ... NAME_N=VAL_1]
 ```

where the optional arguments for the script are as follows:

 * tsv_path - Path to tsv file containing dataset information (default is "benchmarks/gaze.tsv")
 * tracker_type - Tracking algorithm (default is "CSRT", possible values are ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'])
 * use_gpu - Whether to use a gpu (default is False)
 * write_video - Whether to save the processed video via OpenCV (default is False)
 
The results will be saved in `data/videos_gaze_4_TRACKER` directory with "TRACKER" replaced with the chosen tracking algorithm.

## Gaze Object Detector

The following script runs the classifier that receives a video file and the bounding boxes of four objects as input and outputs the index of the objects that is being gazed at by the person in the video. 

 ```bash
python3 scripts/gaze/gaze_detection.py [with NAME_1=VAL_1 ... NAME_N=VAL_1]
 ```

where the optional arguments for the script are as follows:

 * tsv_path - Path to tsv file containing dataset information (default is "benchmarks/gaze.tsv")
 * tracker_type - Tracking algorithm (default is "CSRT", possible values are ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"])
 * write_video - Whether to save the processed video via OpenCV (default is False)
 * use_head - Whether to use the head-landmark-based gaze detection method in the heuristic (default is True)
 * use_body - Whether to use the body-landmark-based gaze detection method in the heuristic (default is True)
 * mix_priority - Which landmark-based gaze ray detection method to prioritise at every frame. Only relevant if previous two arguments are set to True. Possible values are "head", "body", "equal". The "equal" option makes use of both predicted gaze ray (default is "equal")

The results of the experiment are being saved in a directory called `results`.

### Comparing heuristic configurations

In case you're wondering how different configurations of the heuristic method perform, feel free to try them out by running the following scripts.

 ```bash
bash scripts/gaze/try_all_heuristics.sh [log_dir]
 ```

The experimental results will be logged in the given directory (default is "log").

## Credit

This application uses Open Source components. You can find the source code of their open-source projects along with license information below. We acknowledge and are grateful to these developers for their contributions to open source.

Project: Gaze Estimation https://bitbucket.org/phil_dias/gaze-estimation
Copyright (c) Philipe A. Dias (philipe.ambroziodias@marquette.edu)
License (NPOSL-3.0) https://opensource.org/licenses/NPOSL-3.0 

Project: OpenPose https://github.com/CMU-Perceptual-Computing-Lab/openpose
Copyright (c) CMU Perceptual Computing Lab
License https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE 
