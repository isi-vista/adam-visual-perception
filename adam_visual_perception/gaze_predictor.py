from adam_visual_perception import ObjectTracker
from adam_visual_perception.head_gaze_estimator import HeadGazeEstimator
from adam_visual_perception.body_gaze_estimator import BodyGazeEstimator
from adam_visual_perception.utility import *
from collections import OrderedDict
import numpy as np
import math
import cv2
import os
import sys


class GazePredictor:
    """ A class for predicting the object being gazed at. """

    def __init__(
        self,
        tracker_type,
        write_video=False,
        use_head=True,
        use_body=False,
        mix_priority="head",
        use_gpu=True,
    ):
        """
        Parameters
        ----------
        write_video : bool, optional
            Write the resulting OpenCV video
        use_head : bool, optional
            Whether to use the head-landmark-based gaze detection method in the heuristic
        use_body : bool, optional
            Whether to use the body-landmark-based gaze detection method in the heuristic
        mix_priority : str, optional
            Which landmark-based gaze ray detection method to prioritise at every
            frame. Only relevant if previous two arguments are set to True. Possible
            values are "head", "body", "equal". The "equal" option makes use of both
            predicted gaze ray
        use_gpu : bool, optional
            Whether to use a Gpu
        """
        self.write_video = write_video
        self.tracker_type = tracker_type
        self.use_head = use_head
        self.use_body = use_body
        self.mix_priority = mix_priority
        self.use_gpu = use_gpu

        self.head_gaze_estimator = HeadGazeEstimator()
        self.body_gaze_estimator = BodyGazeEstimator()

    def choose_object(self, ray, bboxes, two_lines, length=100000):
        """
        This function makes the prediction on which object to classify as
        being gazed at by the person given the boundary boxes of the objects
        and gaze ray of the person
        """
        # Ray can be either one or two lines (in a list)
        min_dist_facing = length
        min_dist_facing_index = -1

        min_dist_other = length
        min_dist_other_index = -1

        for i, bbox in enumerate(bboxes):
            # Rectagle (x, y), (x + w, y + h)
            x, y = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2]), int(bbox[3])

            # Calculate the distance from the centre
            centre = Point(x + w / 2, y + h / 2)

            if not two_lines:  # Just one line
                ray_origin, ray_dir = ray

                A = Point(ray_origin[0], ray_origin[1])
                B = Point(ray_dir[0], ray_dir[1])
            elif len(ray) == 1:
                ray_origin, ray_dir = ray[0]

                A = Point(ray_origin[0], ray_origin[1])
                B = Point(ray_dir[0], ray_dir[1])
            else:  # Two lines
                r1_origin, r1_dir = ray[0]
                r2_origin, r2_dir = ray[1]

                A = Point(
                    (r1_origin[0] + r2_origin[0]) / 2, (r1_origin[1] + r2_origin[1]) / 2
                )
                B = Point((r1_dir[0] + r2_dir[0]) / 2, (r1_dir[1] + r2_dir[1]) / 2)

            lenAB = math.sqrt((A.x - B.x) ** 2 + (A.y - B.y) ** 2)
            C_x = B.x + (B.x - A.x) / lenAB * length
            C_y = B.y + (B.y - A.y) / lenAB * length

            C = Point(C_x, C_y)

            dist = shortest_dist_to_point(A, C, centre)

            if is_facing_angle(A, C, centre):
                if dist < min_dist_facing:
                    min_dist_facing = dist
                    min_dist_facing_index = i
            else:
                if dist < min_dist_other:
                    min_dist_other = dist
                    min_dist_index = i

        if min_dist_facing < length:
            return min_dist_facing_index, A, C
        else:
            return min_dist_other_index, A, C

    def save_video_file(self, filename, bboxes, rays, prediction):
        """
        Add the gaze ray and tracking objects for each frame of the video
        """
        # Capture the video
        cap = cv2.VideoCapture(filename)
        frame_no = 0

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
                # Initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                par_path = os.path.abspath(os.path.join(filename, os.pardir))

                if self.use_head and not self.use_body:
                    dir_path += "_head"
                elif not self.use_head and self.use_body:
                    dir_path += "_body"
                else:
                    dir_path += "_mixed_" + self.mix_priority

                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)

                video_path = os.path.join(dir_path, os.path.basename(filename))
                writer = cv2.VideoWriter(
                    video_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True
                )

            # Add tracked objects
            rectangles = bboxes[frame_no]
            for i, bbox in enumerate(rectangles):
                x, y = int(bbox[0]), int(bbox[1])
                w, h = int(bbox[2]), int(bbox[3])

                if i + 1 == prediction:
                    color = color_red
                else:
                    color = color_blue
                # Draw bounding boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, 1)

            # Add gaze ray
            if frame_no in rays:
                ray = rays[frame_no]
                p1 = (int(ray[0][0]), int(ray[0][1]))
                p2 = (int(ray[1][0]), int(ray[1][1]))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            key = cv2.waitKey(1) & 0xFF

            writer.write(frame)
            frame_no += 1

        # Cleanup
        cv2.destroyAllWindows()
        writer.release()

    def predict(self, filename, bboxes):
        """
        Predict which of the four objects is being gazed at
        """
        tracker = ObjectTracker(
            tracker_type=self.tracker_type, detect_objects=False, use_gpu=self.use_gpu
        )

        bbox_history, shape = tracker.get_four_bboxes(
            filename, bboxes, save_video=False, return_shape=True
        )

        two_lines = False

        # Case 1: Head On / Body Off
        if self.use_head and not self.use_body:
            gaze_rays = self.head_gaze_estimator.get_gaze_rays(
                filename, bbox_history, show=False
            )
            if len(gaze_rays) == 0:
                gaze_rays = self.body_gaze_estimator.get_gaze_rays(filename, shape)
        # Case 2: Head Off / Body On
        elif not self.use_head and self.use_body:
            gaze_rays = self.body_gaze_estimator.get_gaze_rays(filename, shape)
            if len(gaze_rays) == 0:
                gaze_rays = self.head_gaze_estimator.get_gaze_rays(
                    filename, bbox_history, show=False
                )
        # Case 3: Head On / Body On
        else:
            head_gaze_rays = self.head_gaze_estimator.get_gaze_rays(
                filename, bbox_history, show=False
            )
            body_gaze_rays = self.body_gaze_estimator.get_gaze_rays(filename, shape)
            if self.mix_priority == "head":
                gaze_rays = {**body_gaze_rays, **head_gaze_rays}
            elif self.mix_priority == "body":
                gaze_rays = {**head_gaze_rays, **body_gaze_rays}
            else:  # mix_type == "equal"
                gaze_rays = join_dicts(head_gaze_rays, body_gaze_rays)
                two_lines = True

        # If both are empty, choose randomly
        if len(gaze_rays) == 0:
            return np.random.randint(1, 5)

        preds_per_frame = []

        if self.write_video:
            ray_visual = {}

        gaze_rays_sorted = OrderedDict(sorted(gaze_rays.items()))
        for frame_no, ray in gaze_rays_sorted.items():
            try:
                pred, ray_orig, ray_end = self.choose_object(
                    ray, bbox_history[frame_no], two_lines
                )
                if self.write_video:
                    ray_visual[frame_no] = (ray_orig, ray_end)
                preds_per_frame.append(pred)
            except IndexError:
                continue

        most_common = max(range(4), key=preds_per_frame.count)
        index = 1 + most_common

        if self.write_video:
            self.save_video_file(filename, bbox_history, ray_visual, index)

        return index
