from moviepy.editor import VideoFileClip
from datetime import datetime
import numpy as np
import time
import os


def manage_time(timestamp):
    """
    Given the string representation of a the time using the
    "minutes:seconds[:miliseconds]" representation, returns the number
    of seconds using double precision
    """

    time_strip = timestamp.split(":")
    seconds = int(time_strip[0]) * 60 + int(time_strip[1])

    # Add miliseconds
    if len(time_strip) == 3:
        seconds += int(time_strip[2]) / 60

    return seconds


def preprocess_video(filename, start, end, target_name, audio, codec=None):
    """
    Preprocess an input video by cutting it given start time to end time,
    optionally removing the audio and changing video encoding
    """
    # Load the video file
    clip = VideoFileClip(filename)

    # Calculate start and end points in seconds
    starting_point = manage_time(start)
    end_point = manage_time(end)

    # Resize the video and save the file
    subclip = clip.subclip(starting_point, end_point)
    subclip.write_videofile(target_name, audio=audio, codec=codec)


def concatenate_videos(base="", ext="mp4"):
    stringa = 'ffmpeg -i "concat:'
    elenco_video = glob.glob(base + "*." + ext)
    elenco_file_temp = []
    for f in elenco_video:
        file = "temp" + str(elenco_video.index(f) + 1) + ".ts"
        os.system(
            "ffmpeg -i " + f + " -c copy -bsf:v h264_mp4toannexb -f mpegts " + file
        )
        elenco_file_temp.append(file)
    print(elenco_file_temp)
    for f in elenco_file_temp:
        stringa += f
        if elenco_file_temp.index(f) != len(elenco_file_temp) - 1:
            stringa += "|"
        else:
            stringa += '" -c copy  -bsf:a aac_adtstoasc output.mp4'
    print(stringa)
    os.system(stringa)
