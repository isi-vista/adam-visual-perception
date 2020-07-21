from adam_visual_perception import LandmarkDetector
from adam_visual_perception.utility import *
import numpy as np
import math
import cv2
import os
import sys


class HeadGazeEstimator:
    """ A class for estimating gaze ray from facial landmarks """

    def __init__(self, write_video=False):
        # 3D model points.
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corne
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0),  # Right mouth corner
            ]
        )
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        """
        Parameters
        ----------
        write_video : bool, optional
            Write the resulting OpenCV video
        """

        self.write_video = write_video
        self.landmark_detector = LandmarkDetector(write_video=False)

    def get_gaze_rays(self, filename, bbox_history=None, show=True):
        """
        Get the gaze rays for the given video file
        """
        # Get the landmarks for the entire video
        landmark_map = self.landmark_detector.detect(filename, show=False)

        # Capture the video
        cap = cv2.VideoCapture(filename)
        frame_no = 0

        gaze_angles = {}

        # Loop over the frames from the video stream
        while True:
            success, frame = cap.read()

            if not success:
                if frame_no == 0:
                    print("Failed to read video")
                    sys.exit(1)
                else:
                    break

            if frame_no == 0:
                # Camera internals
                size = frame.shape
                focal_length = size[1]
                center = (size[1] / 2, size[0] / 2)
                camera_matrix = np.array(
                    [
                        [focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1],
                    ],
                    dtype="double",
                )

                if self.write_video:
                    # Initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    par_path = os.path.abspath(os.path.join(filename, os.pardir))
                    dir_path = par_path + "_pnp"
                    if not os.path.isdir(dir_path):
                        os.makedirs(dir_path)
                    video_path = os.path.join(dir_path, os.path.basename(filename))
                    writer = cv2.VideoWriter(
                        video_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True
                    )

            if frame_no in landmark_map:
                # 2D image points.
                image_points = np.array(
                    [
                        landmark_map[frame_no][33],  # Nose tip
                        landmark_map[frame_no][8],  # Chin
                        landmark_map[frame_no][36],  # Left eye left corner
                        landmark_map[frame_no][45],  # Right eye right corne
                        landmark_map[frame_no][48],  # Left Mouth corner
                        landmark_map[frame_no][54],  # Right mouth corner
                    ],
                    dtype="double",
                )

                # We use this to draw a line sticking out of the nose
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.model_points,
                    image_points,
                    camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

                nose_end_point2D, jacobian = cv2.projectPoints(
                    np.array([(0.0, 0.0, 1000.0)]),
                    rotation_vector,
                    translation_vector,
                    camera_matrix,
                    self.dist_coeffs,
                )

                for p in image_points:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 1, (255, 0, 0), -1)

                for p in landmark_map[frame_no]:
                    if p in image_points:
                        continue
                    cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                lenAB = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                length = lenAB * 3
                C_x = int(p2[0] + (p2[0] - p1[0]) / lenAB * length)
                C_y = int(p2[1] + (p2[1] - p1[1]) / lenAB * length)

                cv2.line(frame, p1, (C_x, C_y), (0, 255, 0), 2)

                if bbox_history is not None and (self.write_video or show):
                    bboxes = bbox_history[frame_no]
                    for i, bbox in enumerate(bboxes):
                        x, y = int(bbox[0]), int(bbox[1])
                        w, h = int(bbox[2]), int(bbox[3])

                        cv2.circle(
                            frame, (int(x + w / 2), int(y + h / 2)), 5, (0, 0, 255), -1
                        )

                # Store in the return dictionary
                gaze_angles[frame_no] = (p1, p2)

            # Show the frame if the flag is on
            if show:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

            # Write the video if the flag is on
            if self.write_video:
                writer.write(frame)

            frame_no += 1

        # Cleanup
        cv2.destroyAllWindows()

        if self.write_video:
            writer.release()

        return gaze_angles
