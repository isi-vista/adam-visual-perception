from imutils import face_utils
import imutils
import dlib
import cv2
import sys
import os


class LandmarkDetector:
    """
    A class for detecting face landmarks in the video, such as
    nose, chin, eyes, lips, etc.
    """

    def __init__(
        self,
        shape_predictor="models/shape_predictor_68_face_landmarks.dat",
        write_video=False,
    ):
        """
        Parameters
        ----------
        shape_predictor : str, optional
            Path to face shape predictor model
        write_video : bool, optional
            Write the resulting OpenCV video
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor)
        self.write_video = write_video

    def detect(self, filename, show=True):
        """
        Detect the face in the given video file
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
        frame_no = 0

        landmark_map = {}

        # Loop over the frames from the video stream
        while True:
            success, frame = cap.read()
            if not success:
                if frame_no == 0:
                    print("Failed to read video")
                    sys.exit(1)
                else:
                    break

            if self.write_video:
                # Initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                par_path = os.path.abspath(os.path.join(filename, os.pardir))
                dir_path = par_path + "_landmarks"
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)
                video_path = os.path.join(dir_path, os.path.basename(filename))
                writer = cv2.VideoWriter(
                    video_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True
                )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            rects = self.detector(gray, 0)

            # Loop over the face detections
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                landmark_map[frame_no] = shape
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            if show:
                # show the frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

            # Write the video if the flag is on
            if self.write_video:
                writer.write(frame)

            frame_no += 1

        # Cleanup
        cv2.destroyAllWindows()

        if self.write_video:
            writer.release()

        return landmark_map
