# Installation

## Local installation

Please install the [OpenCV](https://docs.opencv.org/master/d5/de5/tutorial_py_setup_in_windows.html) and [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md) libraries first. Afterwards, clone the repository, cd to the repository directory and run the following command for the package to be installed:

```bash
pip install .
```

Execute the following bash command for downloading pre-trained Dlib and YOLOv3 models.

```bash
bash model_download.sh
```

If you're using Ubuntu, you may find the [docker/Dockerfile](../docker/Dockerfile) instructions useful for local installation.

**Note**: If you have successfully built the docker image and then new changes are pushed to the repo, you will be required to `pull` those changes and `pip install .` again.

**Note**: Please adjust the OpenPose path in the `adam_visual_perception/body_gaze_estimator.py` if it is installed elsewhere than `/openpose`.

## Running via Docker on a Linux cluster

You can conveniently perform experiments on SAGA cluster using the designed docker environment with pre-installed package and prerequisites. To do this, you need to follow the following steps:

```bash
git clone https://github.com/isi-vista/adam-visual-perception
cd adam-visual-perception/docker
./build.sh
./run.sh # Run this from the VNC server to see the OpenCV GUI
```

The [Docker daemon](https://docs.docker.com/config/daemon/) should be running before this can be executed. 
