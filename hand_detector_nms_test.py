import cv2
import numpy as np

from hand_tracker import HandTracker
from non_maximum_suppression import non_max_suppression_fast

ESCAPE_KEY_CODE = 27

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "./palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "./hand_landmark.tflite"
ANCHORS_PATH = "./anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

DRAW_ANCHORS = False
DRAW_DETECTION_BOXES = True
DRAW_BEST_DETECTION_BOX_NMS = True
DRAW_BEST_DETECTION_BOX_MAX_SIZE = True
DRAW_HAND_KEYPOINTS = True





def main():
    cv2.namedWindow(WINDOW)
    capture = cv2.VideoCapture(0)

    if capture.isOpened():
        hasFrame, frame = capture.read()
    else:
        hasFrame = False

    #        8   12  16  20
    #        |   |   |   |
    #        7   11  15  19
    #    4   |   |   |   |
    #    |   6   10  14  18
    #    3   |   |   |   |
    #    |   5---9---13--17
    #    2    \         /
    #     \    \       /
    #      1    \     /
    #       \    \   /
    #        ------0-
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
    ]

    detector = HandTracker(
        PALM_MODEL_PATH,
        LANDMARK_MODEL_PATH,
        ANCHORS_PATH,
        box_shift=0.2,
        box_enlarge=1.3
    )

    while hasFrame:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale = np.array(np.max(frame.shape[:2]) / 256.0)
        padding = [0, 280]

        img_pad, img_norm, pad = detector.preprocess_img(image)
        source, keypoints, debug_info = detector.detect_hand(img_norm)

        if debug_info is not None:
            candidate_detect = debug_info["detection_candidates"]
            candidate_anchors = debug_info["anchor_candidates"]
            selected_candidate_max = debug_info["selected_candidate_max"]

        if DRAW_ANCHORS and debug_info is not None:
            for anchor in candidate_anchors:
                dx, dy = anchor[:2] * 256
                w, h = anchor[2:] * 256 * 0.2 # no idea of 0.2 is the correct size multiplicator
                box = box_from_dimensions(dx - (w/2), dy -(h/2), h, w)
                box *= scale
                box -= padding
                frame = draw_box(frame, box, color=(200, 0, 0))

        if DRAW_DETECTION_BOXES and debug_info is not None:
            for i, detection in enumerate(candidate_detect):
                dx,dy,w,h = detection[:4]
                center_wo_offst = candidate_anchors[i, :2] * 256
                box = box_from_dimensions(dx - (w/2), dy - (h/2), h, w)
                box += center_wo_offst
                box *= scale
                box -= padding
                frame = draw_box(frame, box)

        if DRAW_BEST_DETECTION_BOX_MAX_SIZE and debug_info is not None:
            detection = candidate_detect[selected_candidate_max]
            dx,dy,w,h = detection[:4]
            center_wo_offst = candidate_anchors[selected_candidate_max, :2] * 256
            box = box_from_dimensions(dx - (w/2), dy - (h/2), h, w)
            box += center_wo_offst
            box *= scale
            box -= padding
            frame = draw_box(frame, box, color=(100, 100, 0))

        if DRAW_BEST_DETECTION_BOX_NMS and debug_info is not None:
            boxes = []
            for i, detection in enumerate(candidate_detect):
                dx,dy,w,h = detection[:4]
                center_wo_offst = candidate_anchors[i, :2] * 256
                box = box_from_dimensions(dx - (w/2), dy - (h/2), h, w)
                box += center_wo_offst
                box *= scale
                box -= padding
                boxes.append(box)

            best_boxes = non_max_suppression_fast(np.array(boxes).reshape((-1,8))[:, [0,1,4,5]])
            for box in best_boxes:
                box = from_corners(*box)
                frame = draw_box(frame, box, color=(255, 0, 0))

        if DRAW_HAND_KEYPOINTS and debug_info is not None:
            for i, detection in enumerate(candidate_detect):
                center_wo_offst = candidate_anchors[i, :2] * 256
                hand_key_points = center_wo_offst + detection[4:].reshape(-1, 2)
                for key_point in hand_key_points:
                    key_point *= scale
                    key_point -= padding
                    cv2.circle(frame, tuple(key_point.astype("int")), color=(255, 255, 255), radius=5, thickness=2)



        cv2.imshow(WINDOW, frame)
        hasFrame, frame = capture.read()
        key = cv2.waitKey(20)
        if key == ESCAPE_KEY_CODE:
            break

    capture.release()
    cv2.destroyAllWindows()


def box_from_dimensions(dx, dy, h, w):
    box = [[dx, dy], [dx + w, dy],
           [dx + w, dy + h], [dx, dy + h]]
    return box

def from_corners(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

def draw_box(image, box, thickness=2, color=(0, 255, 0)):
    image = image.copy()
    if box is None or len(box) < 4:
        return
    for i in range(0, len(box)):
        i_inc_wrapped = (i + 1) % len(box)
        start_x, start_y = box[i]
        end_x, end_y = box[i_inc_wrapped]
        cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color, thickness)
    return image


if __name__ == '__main__':
    main()