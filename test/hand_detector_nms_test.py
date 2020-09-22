import cv2
import numpy as np

from src.hand_tracker import HandTracker

ESCAPE_KEY_CODE = 27

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"


DRAW_ANCHORS = True
DRAW_DETECTION_BOXES = True
DRAW_BEST_DETECTION_BOX_NMS = True
DRAW_HAND_KEYPOINTS = True


def main():
    cv2.namedWindow(WINDOW)
    capture = cv2.VideoCapture(0)

    if capture.isOpened():
        hasFrame, frame = capture.read()
    else:
        hasFrame = False


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
        source, keypoints, debug_info = detector.detect_hand(img_norm,0)
        source2,keypoints2,debug_info2 = detector.detect_hand(img_norm,1)
        if debug_info is not None:
            candidate_detect = debug_info["detection_candidates"]
            candidate_anchors = debug_info["anchor_candidates"]
            selected_box_id = debug_info["selected_box_id"]
            print(selected_box_id)
        if debug_info2 is not None:
            candidate_detect2 = debug_info2["detection_canditates"]
            candidate_anchors2 = debug_info2["anchor_candidates"]
            selected_box_id2= debug_info2["selected_box_id"]
            print(selected_box_id2)

        if DRAW_ANCHORS and debug_info is not None:
            for anchor in candidate_anchors:
                dx, dy = anchor[:2] * 256
                w, h = anchor[2:] * 256 * 0.2  # no idea of 0.2 is the correct size multiplication
                box = box_from_dimensions(dx - (w/2), dy - (h/2), h, w)
                box *= scale
                box -= padding
                frame = draw_box(frame, box, color=(200, 0, 0))
        if DRAW_ANCHORS and debug_info2 is not None:
            for anchor in candidate_anchors2:
                dx2,dy2 = anchor[:2] * 256
                w2,h2 = anchor[2:] * 256 *0.2
                box2=box_from_dimensions(dx2 - (w2/2),dy2-(h2/2),h2,w2)
                box2*=scale
                box2-=padding
                frame = draw_box(frame,box2,color=(200,0,0))
        if DRAW_DETECTION_BOXES and debug_info is not None:
            for i, detection in enumerate(candidate_detect):
                dx,dy,w,h = detection[:4]
                center_wo_offst = candidate_anchors[i, :2] * 256
                box = box_from_dimensions(dx - (w/2), dy - (h/2), h, w)
                box += center_wo_offst
                box *= scale
                box -= padding
                frame = draw_box(frame, box)
        if DRAW_DETECTION_BOXES and debug_info2 is not None:
            for i,detection in enumerate(candidate_detect2):
                dx2,dy2,w2,h2 = detection[:4]
                center_wo_offset2=candidate_anchors2[i,:2]*256
                box2= box_from_dimensions(dx2 - (w2/2),dy2-(h2/2),h2,w2)
                box2 += center_wo_offset2
                box2 *=scale
                box2 -=padding
                frame=draw_box(frame,box2)
        if DRAW_HAND_KEYPOINTS and debug_info is not None:
            detection = candidate_detect[selected_box_id]
            center_wo_offst = candidate_anchors[i, :2] * 256
            hand_key_points = center_wo_offst + detection[4:].reshape(-1, 2)
            for key_point in hand_key_points:
                key_point *= scale
                key_point -= padding
                cv2.circle(frame, tuple(key_point.astype("int")), color=(255, 255, 255), radius=5, thickness=2)
        if DRAW_HAND_KEYPOINTS and debug_info2 is not None:
            detection2 = candidate_detect2[selected_box_id2]
            center_wo_offset2 = candidate_anchors2[i,:2]*256
            hand_key_points2 = center_wo_offset2 + detection2[4:].reshape(-1,2)
            for key_point in hand_key_points2:
                key_point *=scale
                key_point -=padding
                cv2.circle(frame,tuple(key_point.astype("int")),color=(255,255,255),radius=5,thickness=2)


        if DRAW_BEST_DETECTION_BOX_NMS and debug_info is not None:
            detection = candidate_detect[selected_box_id]
            dx, dy, w, h = detection[:4]
            center_wo_offst = candidate_anchors[selected_box_id, :2] * 256
            box = box_from_dimensions(dx - (w / 2), dy - (h / 2), h, w)
            box += center_wo_offst
            box *= scale
            box -= padding
            frame = draw_box(frame, box, color=(0, 0, 255))
        if DRAW_BEST_DETECTION_BOX_NMS and debug_info2 is not None:
            detection2 = candidate_detect2[selected_box_id2]
            dx2,dy2,w2,h2 = detection2[:4]
            center_wo_offset2 = candidate_anchors2[selected_box_id2,:2]*256
            box = box_from_dimensions(dx2,(w2/2),dy2 ( h2/2),h2,w2)
            box +=center_wo_offset2
            box*=scale
            box-=padding

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