#!/bin/bash

# This script launches all possible variations of the heuristic method for
# gaze detection and stores the results in the log directory.

if [[ -z "$1" ]] ; then
    log_dir='log'
else
    log_dir=$1
fi

mkdir -p $log_dir

python3 scripts/gaze/gaze_detection.py with use_head=True use_body=False > $log_dir/head.txt
python3 scripts/gaze/gaze_detection.py with use_head=False use_body=True > $log_dir/body.txt
python3 scripts/gaze/gaze_detection.py with use_head=True use_body=True mix_priority=head > $log_dir/mixed_head.txt
python3 scripts/gaze/gaze_detection.py with use_head=True use_body=True mix_priority=body > $log_dir/mixed_body.txt
python3 scripts/gaze/gaze_detection.py with use_head=True use_body=True mix_priority=equal > $log_dir/mixed_equal.txt
