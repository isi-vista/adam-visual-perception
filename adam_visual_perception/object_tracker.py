from adam_visual_perception.utility import *
import numpy as np
import imutils
import time
import cv2
import sys
import os


class ObjectTracker:
    """
    A class for object tracking and detection
    """

    def __init__(
        self,
        tracker_type="BOOSTING",
        target_classes=["person"],
        thresh_conf=0.5,
        thresh_nms=0.3,
        yolo_path="yolo",  # Path to yolo dir
        use_gpu=True,
        predict_time=False,
        detect_objects=True,
        moving_thresh=3,
        print_move_info=False,
        write_video=False,
    ):
        """
        Parameters
        ----------
        tracker_type : str, optional
            The type of the tracking method
        target_classes : list, optional
            The classes of objects that can move objects
        thresh_conf : float, optional
            Detection confidence threshold for YOLO
        thresh_nms : float, optional
            Threshold for non-maxima suppression
        use_gpu : bool, optional
            Whether to use gpu
        predict_time : bool, optional
            Predict how much time will the experimemnt take
        detect_objects : bool, optional
            Whether to perform detection alongside tracking
        moving_thresh : int, optional
            How much distance is considered moving
        print_move_info : bool, optional
            Display whether object is moving in OpenCV
        write_video : bool, optional
            Write the resulting OpenCV video
        """
        self.tracker_type = tracker_type
        self.thresh_conf = thresh_conf
        self.thresh_nms = thresh_nms
        self.target_classes = target_classes
        self.predict_time = predict_time
        self.detect_objects = detect_objects
        self.moving_thresh = moving_thresh
        self.print_move_info = print_move_info
        self.write_video = write_video

        if self.detect_objects:
            # Load object label names (from Coco)
            labels_path = os.path.join(yolo_path, "coco.names")
            self.labels = open(labels_path).read().strip().split("\n")

            # derive the paths to the YOLO weights and model configuration
            weights = os.path.join(yolo_path, "yolov3.weights")
            config = os.path.join(yolo_path, "yolov3.cfg")

            print("[INFO] loading YOLO from disk...")
            self.net = cv2.dnn.readNetFromDarknet(config, weights)
            if use_gpu:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            ln = self.net.getLayerNames()
            self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def get_single_prediction_time(self, frame):
        """
        Predict time necessary for performing object detection on the input video
        """
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self.ln)
        end = time.time()
        return end - start

    def object_detection(self, frame):
        """
        Recognise persons or other "external forces in the given frame in the frame
        """
        # Construct a blob from the input frame and perform a forward pass of YOLO
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)

        # initialize our lists of detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        classIDs = []
        H, W = frame.shape[:2]

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out desired classes and weak predictions
                if (
                    self.labels[classID] in self.target_classes
                    and confidence > self.thresh_conf
                ):
                    # Scale the bbox coords relative to the size of the image
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Non-maxima suppression to suppress weak and overlapping bboxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.thresh_conf, self.thresh_nms)

        objects = []
        if len(idxs) == 0:
            return []

        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            self.labels[classIDs[i]]
            confidences[i]
            objects.append(
                ((x, y), (x + w, y + h), self.labels[classIDs[i]], confidences[i])
            )

        # Sort objects by their position on the screen
        return sorted(objects, key=lambda x: x[0])

    def get_bboxes(self, filename, bbox):
        """
        Get the bounding boxes of the target object and detected object for all frames
        of the input video
        """
        # Capture the video
        cap = cv2.VideoCapture(filename)

        # Total number of frames
        prop = (
            cv2.cv.CV_CAP_PROP_FRAME_COUNT
            if imutils.is_cv2()
            else cv2.CAP_PROP_FRAME_COUNT
        )
        total = int(cap.get(prop))

        # Define the tracker
        tracker = createTrackerByName(self.tracker_type)

        frame_no = 0
        target_bboxes = []
        if self.detect_objects:
            object_bboxes = {}

        while True:
            # Read the next frame
            success, frame = cap.read()

            if frame_no == 0:
                if not success:
                    print("Failed to read video")
                    sys.exit(1)

                ok = tracker.init(frame, bbox)

                if self.write_video:
                    # Initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    par_path = os.path.abspath(os.path.join(filename, os.pardir))
                    dir_path = par_path + "_tracking"
                    if not os.path.isdir(dir_path):
                        os.makedirs(dir_path)
                    video_path = os.path.join(dir_path, os.path.basename(filename))
                    writer = cv2.VideoWriter(
                        video_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True
                    )
                if self.predict_time:
                    # this isn't quite accurate
                    elap = self.get_single_prediction_time(frame)
                    print("[INFO] single prediction took {:.2f}s".format(elap))
                    print(
                        "[INFO] estimated total time to finish: {:.2f}s".format(
                            elap * total
                        )
                    )
            else:
                # Break if return value of read is False
                if not success:
                    break

                # Update tracker
                ok, bbox = tracker.update(frame)

            if ok:
                # Tracking success
                x, y = int(bbox[0]), int(bbox[1])
                w, h = int(bbox[2]), int(bbox[3])

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_red, 2, 1)
                target_bboxes.append(((x, y), (x + w, y + h)))

                if self.print_move_info:
                    curr, _ = target_bboxes[-1]
                    prev, _ = target_bboxes[-2]

                    dist = get_distance(curr[0], curr[1], prev[0], prev[1])
                    if dist < self.moving_thresh:
                        cv2.putText(
                            frame,
                            "NOT MOVING",
                            (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color_green,
                            2,
                        )
                    else:
                        cv2.putText(
                            frame,
                            "MOVING",
                            (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color_green,
                            2,
                        )
            else:
                if frame_no == 0:
                    print("Cannot Initialize the object tracker")
                    sys.exit(1)
                else:
                    cv2.putText(
                        frame,
                        "Tracking failure detected",
                        (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        color_red,
                        2,
                    )

            if self.detect_objects:
                object_bboxes[frame_no] = []

                objects = self.object_detection(frame)
                # draw a bounding box rectangle and label
                for o in objects:
                    (x1, y1), (x2, y2), class_name, conf = o

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text = "{}: {:.4f}".format(class_name, conf)
                    cv2.putText(
                        frame,
                        text,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color_blue,
                        2,
                    )

                    object_bboxes[frame_no].append(((x1, y1), (x2, y2)))

            # Display result
            cv2.imshow("Tracking", frame)

            # Write the video if the flag is on
            if self.write_video:
                writer.write(frame)

            frame_no += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if self.write_video:
            writer.release()
        cv2.destroyAllWindows()

        if self.detect_objects:
            return target_bboxes, object_bboxes
        else:
            return target_bboxes

    def get_overlaps(self, target_hist, object_hist, consider_size=False):
        """
        Get information on overlaps between the target object and detected objects
        """
        overlaps = []
        num_frame = len(target_hist)

        for i in range(num_frame):
            # if no detected object
            if len(object_hist[i]) == 0:
                overlaps.append(False)
                continue

            # check overlaps between target and all objects
            overlap = False
            target = target_hist[i]
            for o in object_hist[i]:
                l1 = Point(target[0][0], target[0][1])
                r1 = Point(target[1][0], target[1][1])
                l2 = Point(o[0][0], o[0][1])
                r2 = Point(o[1][0], o[1][1])

                intersection = check_overlap(l1, r1, l2, r2)

                if not consider_size:
                    overlap = overlap or intersection
                else:
                    bigger = (r1.x - l1.x) * (r1.y - l1.y) < (r2.x - l2.x) * (
                        r2.y - l2.y
                    )
                    overlap = overlap or (intersection and bigger)

            overlaps.append(overlap)

        return overlaps

    def get_movement(self, target_bboxes):
        """
        Return a list stating whether the object moved during each frame.
        This isn't ideal because the camera is moving.
        """
        movement = []
        for i in range(len(target_bboxes)):
            if i == 0:
                movement.append(False)
            else:
                # Only considering upper left corner of the bbox
                curr, _ = target_bboxes[i]
                prev, _ = target_bboxes[i - 1]

                dist = get_distance(curr[0], curr[1], prev[0], prev[1])
                if dist < self.moving_thresh:
                    movement.append(False)
                else:
                    movement.append(True)

        return movement

    def movement_heuristic(self, target_hist, object_hist):
        """
        Determine whether target is moving by itself or not based on the
        bounding box history of the target and detected objects.
        """

        num_frame = len(target_hist)

        # if no detected object, then no external forces
        if sum(len(val) for key, val in object_hist.items()) == 0:
            return 0

        # if no overlaps, then no external forces
        overlaps = self.get_overlaps(target_hist, object_hist, consider_size=True)
        num_overlaps = len([o for o in overlaps if o])
        if num_overlaps < 0.05 * num_frame:
            return 0

        # Frames where the target has moved
        movement = self.get_movement(target_hist)

        # Number of movement
        num_moved = len([m for m in movement if m])
        # Number of movements which coincides with overlaps
        num_moved_and_overlap = len([i and j for i, j in zip(overlaps, movement)])

        # If moved (mostly) when overlapped with a person, then moved by external force
        if num_moved_and_overlap > num_moved * 0.75:
            return 1

        return 0

    def predict(self, filename, bbox):
        target_bboxes, object_bboxes = self.get_bboxes(filename, bbox)

        assert len(target_bboxes) == len(
            object_bboxes
        ), "target and object bboxes have different lengths"

        return self.movement_heuristic(target_bboxes, object_bboxes)

    def get_four_bboxes(
        self, filename, bboxes, label=None, save_video=False, return_shape=False
    ):
        """
        Get the bounding boxes of the four selected objects
        """
        # Capture the video
        cap = cv2.VideoCapture(filename)

        # Define the tracker
        t1 = createTrackerByName(self.tracker_type)
        t2 = createTrackerByName(self.tracker_type)
        t3 = createTrackerByName(self.tracker_type)
        t4 = createTrackerByName(self.tracker_type)

        trackers = [t1, t2, t3, t4]

        frame_no = 0
        bbox_hist = []

        while True:
            # Read the next frame
            success, frame = cap.read()

            if frame_no == 0:
                if not success:
                    print("Failed to read video")
                    sys.exit(1)

                if return_shape:
                    ret_shape = frame.shape

                for tracker, bbox in zip(trackers, bboxes):
                    tracker.init(frame, bbox)

                if self.write_video:
                    # Initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    par_path = os.path.abspath(os.path.join(filename, os.pardir))
                    dir_path = par_path + "_4_" + self.tracker_type
                    if not os.path.isdir(dir_path):
                        os.makedirs(dir_path)
                    video_path = os.path.join(dir_path, os.path.basename(filename))
                    writer = cv2.VideoWriter(
                        video_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True
                    )
            else:
                # Break if return value of read is False
                if not success:
                    break

            # four bboxes for the current frame
            frame_bboxes = []
            for i, tracker in enumerate(trackers):
                # Update tracker
                _, bbox = tracker.update(frame)

                x, y = int(bbox[0]), int(bbox[1])
                w, h = int(bbox[2]), int(bbox[3])

                # Use red for target object and blue for others
                if label is not None and i + 1 == label:
                    color = color_red
                else:
                    color = color_blue

                # Draw bounding boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, 1)
                frame_bboxes.append(bbox)

            bbox_hist.append(frame_bboxes)

            # Write the video if the flag is on
            if self.write_video:
                writer.write(frame)

            frame_no += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Cleanup
        cap.release()
        if self.write_video:
            writer.release()
        cv2.destroyAllWindows()

        if return_shape:
            return bbox_hist, ret_shape

        return bbox_hist
