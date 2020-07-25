# Motion Cause Detection

### Abstract

Our learning process depends on having a fairly rich (though developmentally plausible) input representation. It could reasonably be asked whether the features of such a representation could be derived by a computational system from raw sensory input. To explore the applicability of approaches developed in ADAM to naturally occurring data, we are investigating whether we could extract the `self-moving` feature from videos; this feature distinguished objects which move under their own power from those which are moved by some external force.

### Task

Given a video segment and an object in the video, determine whether it is moving under its own power or not (i.e. whether it should have the `self-moving` property).

### Method

Key to our approach is two subproblems of computer vision which have been studied extensively during past decades.

First, we perform **object tracking** with respect to the target object, i.e., identifying where the target object is at any point in time during the entire video segment. Since the bounding box of the target object is provided for the first frame, our goal is to move the bounding box (and change its size if necessary) during each subsequent frame. We make use of real-time object tracking based on the on-line version of the [AdaBoost](http://www.bmva.org/bmvc/2006/papers/033.pdf) algorithm in [OpenCV](https://opencv.org/) to achieve this.

Second, we rely on **object detection** techniques to tell us what else is in each of the frames in our video segment. Specifically, what we are interested in is whether there are humans present in the frame and where are they located. We use a popular object detection method called [YOLOv3](https://pjreddie.com/darknet/yolo/) that provides us with the bounding boxes of the persons for each frame. Object detection is followed by on-maxima suppression to suppress weak and overlapping bounding boxes of the detected objects.

Having bounding box sequences of the target object and humans for each frame (there may also be no human at all), our classifier makes its predictions based on how the two sequences interact. Specifically, if there was no overlap between the target object and persons, then we predict that the target is moving by itself. If the target has moved mostly when overlapping with a person, then there is a good chance that an external force is causing the movement.

### Results

Our method archives 93.3% accuracy on our dataset comprised of 60 video segments (56 out of 60).

To give an intuition on how our method works, we provide a [video](https://youtu.be/5AMeJr-7lJg) that shows the bounding box sequence of the target object and humans at each frame for several examples. 

# Scripts: Step-by-step guide

Please follow the following steps to replicate the results of our experiments:

- [Data Preprocessing](#data-preprocessing): before running our methods on video clips, we preprocess them. Specifically, we cut the video segments of our interest, make the size smaller, etc,
- [Target Object Labeling](#target-object-labeling): once video clips are ready, we can run a script that will allow users to identify the target object of the experiment,
- [Testing Object Tracking](#testing-object-tracking): in case you want to take a quick look into how the tracking methods work for the target object,
- [Motion Cause Detector](#motion-cause-detector): classifying the target of the videos either being moved by itself or due to an external force.

## Data Preprocessing

The following script loads the videos, removes the audio track, cuts the videos from the start point to endpoint for given dataset fragments, and saves them in a new directory. A new `tsv` file is being generated with corresponding filenames and ground truth labels.

```bash
python3 scripts/motion/preprocess.py [with NAME_1=VAL_1 ... NAME_N=VAL_1]
```

where the optional arguments for the script are as follows:

 * data_dir - Directory path to raw videos (default is "/path/to/raw/videos")
 * tsv_path - Path to tsv file containing dataset information (default is "benchmarks/motion_raw.tsv")
 * target_dir - Where to save the preprocessed videos (default is "data/videos_motion")
 * target_tsv - Where to write information on preprocessed vidoes (default is "benchmarks/motion.tsv")
 * base_name - Base name of preprocessed vidoes (default is "video")
 * audio - Whether to include the audio in the video files (default is False)

**Note**: You don't need to perform this step unless you have extended the benchmark data.

 ## Target Object Labeling

The following script, given a .tsv file containing video files, opens them one after another, prompts the user to draw a bounding box around the object that will be serving as a target in that video. The bounding boxes will be written in the same .tsv file. If the given .tsv file already has some of the inputs labelled, the user will be prompted to only label non-labelled inputs.

 ```bash
python3 scripts/motion/label_objects.py [with tsv_path=VAL]
 ```

where the optional arguments for the script are as follows:

 * tsv_path - Path to tsv file containing dataset information (default is "benchmarks/motion.tsv"

**Note**: You don't need to perform this step unless you have extended the benchmark data.

## Testing Object Tracking

The following script receives a .tsv file as input which has already been labelled (see the previous step) and runs the given object tracking algorithm on all videos. 

 ```bash
python3 scripts/motion/object_tracking.py [with NAME_1=VAL_1 ... NAME_N=VAL_1]
 ```

where the optional arguments for the script are as follows:

 * tsv_path - Path to tsv file containing dataset information (default is "benchmarks/motion.tsv")
 * tracker_type - Tracking algorithm (default is "BOOSTING", possible values are ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'])
 * use_gpu - Whether to use a gpu (default is False)
 * write_video - Whether to save the processed video via OpenCV (default is False)

## Motion Cause Detector

The following script runs the binary classifier that receives a video file and the bounding box of the target object as input and outputs whether it was moved by itself or due to an external force. 

 ```bash
python3 scripts/motion/cause_detection.py [with NAME_1=VAL_1 ... NAME_N=VAL_1]
 ```

where the optional arguments for the script are as follows:

 * tsv_path - Path to tsv file containing dataset information (default is "benchmarks/motion.tsv")
 * tracker_type - Tracking algorithm (default is "BOOSTING", possible values are ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"])
 * thresh_conf - Confidence threshold for object detection of YOLO (default is 0.85)
 * use_gpu - Whether to use a gpu (default is True)
 * write_video - Whether to save the processed video via OpenCV (default is True)
 * yolo_path - Path to yolo weights and config files (default is "yolo")

The results of the experiment are being saved in a directory called `results`.
