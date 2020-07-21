#!/bin/bash

# Download pre-trained Dlib face prediction model
wget https://github.com/AKSHAYUBHAT/TensorFace/raw/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
mv shape_predictor_68_face_landmarks.dat models/

# Download the weights of YOLOv3 model
wget https://pjreddie.com/media/files/yolov3.weights
mv yolov3.weights yolo/
