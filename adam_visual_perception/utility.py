from collections import namedtuple
import scipy.io as sio
import numpy as np
import cv2
import math
import json


# Defining a 2D point class
Point = namedtuple("Point", ["x", "y"])


def check_overlap(l1, r1, l2, r2):
    """
    Check the overlap between two points
    """
    # If one rectangle is on left side of other
    if r1.x < l2.x or l1.x > r2.x:
        return False

    # If one rectangle is above other
    if r1.y < l2.y or l1.y > r2.y:
        return False

    return True


def merge_rectangles(l1, r1, l2, r2):
    """
    Merge two rectangles
    """
    l = Point(min(l1.x, l2.x), min(l1.y, l2.y))
    r = Point(max(r1.x, r2.x), max(r1.y, r2.y))
    return l, r


def add_bounding_box(existing_boxes, l, r):
    """
    This method iteratively adds bounding boxes toghether.
    Should two boxes overlap, they are automatically merged together
    """
    if not existing_boxes:
        existing_boxes.append((l, r))
        return existing_boxes

    merged = False
    new_l, new_r = None, None

    for l1, r1 in existing_boxes:
        if check_overlap(l1, r1, l, r):
            new_l, new_r = merge_rectangles(l1, r1, l, r)
            existing_boxes.remove((l1, r1))
            merged = True
            break

    if merged:
        existing_boxes = add_bounding_box(existing_boxes, new_l, new_r)
    else:
        existing_boxes.append((l, r))

    return existing_boxes


def draw_bbox(filename, txt=None):
    """
    This function propmts the user to draw a bounding box on the
    first frame of the given video
    """
    cap = cv2.VideoCapture(filename)

    # Read first frame
    success, frame = cap.read()

    # quit if unable to read the video file
    if not success:
        print("Failed to read video")
        sys.exit(1)

    if txt:
        cv2.putText(frame, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_green, 2)

    # Define an initial bounding box
    bbox = cv2.selectROI(frame, False)

    cap.release()
    cv2.destroyAllWindows()

    return bbox


def get_distance(x1, y1, x2, y2):
    """
    Euclidean distance between two points
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


trackerTypes = [
    "BOOSTING",
    "MIL",
    "KCF",
    "TLD",
    "MEDIANFLOW",
    "GOTURN",
    "MOSSE",
    "CSRT",
]
color_red = (255, 0, 0)
color_green = (0, 255, 0)
color_blue = (0, 0, 255)


def createTrackerByName(trackerType):
    """
    Create an OpenCV tracker given the type
    """
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print("Incorrect tracker name")
        print("Available trackers are:")
        for t in trackerTypes:
            print(t)

    return tracker


def sq_shortest_dist_to_point(a, b, other_point):
    dx = b.x - a.x
    dy = b.y - a.y
    dr2 = float(dx ** 2 + dy ** 2)

    lerp = ((other_point.x - a.x) * dx + (other_point.y - a.y) * dy) / dr2
    if lerp < 0:
        lerp = 0
    elif lerp > 1:
        lerp = 1

    x = lerp * dx + a.x
    y = lerp * dy + a.y

    _dx = x - other_point.x
    _dy = y - other_point.y
    square_dist = _dx ** 2 + _dy ** 2
    return square_dist


def shortest_dist_to_point(a, b, other_point):
    return math.sqrt(sq_shortest_dist_to_point(a, b, other_point))


def is_facing_angle(a, b, other_point):
    dx = b.x - a.x
    dy = b.y - a.y
    dr2 = float(dx ** 2 + dy ** 2)

    lerp = ((other_point.x - a.x) * dx + (other_point.y - a.y) * dy) / dr2
    return lerp >= 0


# draw arrow illustrating gaze direction on the image
def draw_on_img(img, centr, id_, res):
    res[0] *= img.shape[0]
    res[1] *= img.shape[1]

    norm1 = res / np.linalg.norm(res)
    norm1[0] *= img.shape[0] * 0.15
    norm1[1] *= img.shape[0] * 0.15

    point = centr + norm1

    # result = cv2.circle(img, (int(point[0]),int(point[1])), 5, (0,0,255), 2)
    result = cv2.arrowedLine(
        img,
        (int(centr[0]), int(centr[1])),
        (int(point[0]), int(point[1])),
        (0, 0, 0),
        thickness=3,
        tipLength=0.2,
    )
    result = cv2.arrowedLine(
        result,
        (int(centr[0]), int(centr[1])),
        (int(point[0]), int(point[1])),
        (0, 0, 255),
        thickness=2,
        tipLength=0.2,
    )

    txtPos = [int(centr[0]), int(centr[1]) - 10]
    result = cv2.putText(
        result,
        str(id_),
        tuple(txtPos),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def merge_values(val1, val2):
    if val1 is None:
        return [val2]
    elif val2 is None:
        return [val1]
    else:
        return [val1, val2]


# The functions below are copied from https://bitbucket.org/phil_dias/gaze-estimation


def join_dicts(x, y):
    return {key: merge_values(x.get(key), y.get(key)) for key in set(x).union(y)}


def load_poses_from_json(json_filename):

    with open(json_filename) as data_file:
        loaded = json.load(data_file)
        poses, conf = json_to_poses(loaded)

    if len(poses) != 1:
        return None, None
    else:
        return poses, conf


def load_many_poses_from_json(json_filename):

    with open(json_filename) as data_file:
        loaded = json.load(data_file)
        poses, conf = json_to_poses(loaded)

    return poses, conf


def json_to_poses(js_data):

    poses = []
    confidences = []

    for arr in js_data["people"]:
        confidences.append(arr["pose_keypoints_2d"][2::3])
        arr = np.delete(arr["pose_keypoints_2d"], slice(2, None, 3))
        poses.append(list(zip(arr[::2], arr[1::2])))

    return poses, confidences


def compute_head_features(pose, conf):

    joints = [0, 15, 16, 17, 18]

    n_joints_set = [pose[joint] for joint in joints if joint_set(pose[joint])]

    if len(n_joints_set) < 1:
        return None, None

    centroid = compute_centroid(n_joints_set)

    max_dist = max([dist_2D(j, centroid) for j in n_joints_set])

    new_repr = [(np.array(pose[joint]) - np.array(centroid)) for joint in joints]

    result = []

    for i in range(0, 5):

        if joint_set(pose[joints[i]]):
            result.append([new_repr[i][0] / max_dist, new_repr[i][1] / max_dist])
        else:
            result.append([0, 0])

    flat_list = [item for sublist in result for item in sublist]

    conf_list = []

    for j in joints:
        conf_list.append(conf[j])

    return flat_list, conf_list, centroid


def compute_body_features(pose, conf):

    joints = [0, 15, 16, 17, 18]
    alljoints = range(0, 25)

    n_joints_set = [pose[joint] for joint in joints if joint_set(pose[joint])]

    if len(n_joints_set) < 1:
        return None, None

    centroid = compute_centroid(n_joints_set)

    n_joints_set = [pose[joint] for joint in alljoints if joint_set(pose[joint])]

    max_dist = max([dist_2D(j, centroid) for j in n_joints_set])

    new_repr = [(np.array(pose[joint]) - np.array(centroid)) for joint in alljoints]

    result = []

    for i in range(0, 25):

        if joint_set(pose[i]):
            result.append([new_repr[i][0] / max_dist, new_repr[i][1] / max_dist])
        else:
            result.append([0, 0])

    flat_list = [item for sublist in result for item in sublist]

    for j in alljoints:
        flat_list.append(conf[j])

    return flat_list, centroid


def compute_centroid(points):

    mean_x = np.mean([p[0] for p in points])
    mean_y = np.mean([p[1] for p in points])

    return [mean_x, mean_y]


def joint_set(p):

    return p[0] != 0.0 or p[1] != 0.0


def dist_2D(p1, p2):

    # print(p1)
    # print(p2)

    p1 = np.array(p1)
    p2 = np.array(p2)

    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    return np.sqrt(squared_dist)


def compute_head_centroid(pose):

    joints = [0, 15, 16, 17, 18]

    n_joints_set = [pose[joint] for joint in joints if joint_set(pose[joint])]

    if len(n_joints_set) < 2:
        return None

    centroid = compute_centroid(n_joints_set)

    r
