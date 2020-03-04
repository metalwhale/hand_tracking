import csv
import cv2
import numpy as np
import tensorflow as tf

from src.non_maximum_suppression import non_max_suppression_fast


class HandTracker():
    r"""
    Class to use Google's Mediapipe HandTracking pipeline from Python.
    So far only detection of a single hand is supported.
    Any any image size and aspect ratio supported.

    Args:
        palm_model: path to the palm_detection.tflite
        joint_model: path to the hand_landmark.tflite
        anchors_path: path to the csv containing SSD anchors
    Ourput:
        (21,2) array of hand joints.
    Examples::
        >>> det = HandTracker(path1, path2, path3)
        >>> input_img = np.random.randint(0,255, 256*256*3).reshape(256,256,3)
        >>> keypoints, bbox = det(input_img)
    """

    def __init__(self, palm_model, joint_model, anchors_path,
                box_enlarge=1.5, box_shift=0.2):
        self.box_shift = box_shift
        self.box_enlarge = box_enlarge

        self.interp_palm = tf.lite.Interpreter(palm_model)
        self.interp_palm.allocate_tensors()
        self.interp_joint = tf.lite.Interpreter(joint_model)
        self.interp_joint.allocate_tensors()

        # reading the SSD anchors
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]
        # reading tflite model paramteres
        output_details = self.interp_palm.get_output_details()
        input_details = self.interp_palm.get_input_details()

        self.in_idx = input_details[0]['index']
        self.out_reg_idx = output_details[0]['index']
        self.out_clf_idx = output_details[1]['index']

        self.in_idx_joint = self.interp_joint.get_input_details()[0]['index']
        self.out_idx_joint = self.interp_joint.get_output_details()[0]['index']

        # 90° rotation matrix used to create the alignment trianlge
        self.R90 = np.r_[[[0,1],[-1,0]]]

        # trianlge target coordinates used to move the detected hand
        # into the right position
        self._target_triangle = np.float32([
                        [128, 128],
                        [128,   0],
                        [  0, 128]
                    ])
        self._target_box = np.float32([
                        [  0,   0, 1],
                        [256,   0, 1],
                        [256, 256, 1],
                        [  0, 256, 1],
                    ])

    def _get_triangle(self, kp0, kp2, dist=1):
        """get a triangle used to calculate Affine transformation matrix"""

        dir_v = kp2 - kp0
        dir_v /= np.linalg.norm(dir_v)

        dir_v_r = dir_v @ self.R90.T
        return np.float32([kp2, kp2+dir_v*dist, kp2 + dir_v_r*dist])

    @staticmethod
    def _triangle_to_bbox(source):
        # plain old vector arithmetics
        bbox = np.c_[
            [source[2] - source[0] + source[1]],
            [source[1] + source[0] - source[2]],
            [3 * source[0] - source[1] - source[2]],
            [source[2] - source[1] + source[0]],
        ].reshape(-1,2)
        return bbox

    @staticmethod
    def _im_normalize(img):
         return np.ascontiguousarray(
             2 * ((img / 255) - 0.5
        ).astype('float32'))

    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x) )

    @staticmethod
    def _pad1(x):
        return np.pad(x, ((0,0),(0,1)), constant_values=1, mode='constant')


    def predict_joints(self, img_norm):
        self.interp_joint.set_tensor(
            self.in_idx_joint, img_norm.reshape(1,256,256,3))
        self.interp_joint.invoke()

        joints = self.interp_joint.get_tensor(self.out_idx_joint)
        return joints.reshape(-1,2)

    def detect_hand(self, img_norm):
        assert -1 <= img_norm.min() and img_norm.max() <= 1,\
        "img_norm should be in range [-1, 1]"
        assert img_norm.shape == (256, 256, 3),\
        "img_norm shape must be (256, 256, 3)"

        # predict hand location and 7 initial landmarks
        self.interp_palm.set_tensor(self.in_idx, img_norm[None])
        self.interp_palm.invoke()

        """
        out_reg shape is [number of anchors, 18]
        Second dimension 0 - 4 are bounding box offset, width and height: dx, dy, w ,h
        Second dimension 4 - 18 are 7 hand keypoint x and y coordinates: x1,y1,x2,y2,...x7,y7
        """
        out_reg = self.interp_palm.get_tensor(self.out_reg_idx)[0]
        """
        out_clf shape is [number of anchors]
        it is the classification score if there is a hand for each anchor box
        """
        out_clf = self.interp_palm.get_tensor(self.out_clf_idx)[0,:,0]

        # finding the best prediction
        probabilities = self._sigm(out_clf)
        detecion_mask = probabilities > 0.5
        candidate_detect = out_reg[detecion_mask]
        candidate_anchors = self.anchors[detecion_mask]
        probabilities = probabilities[detecion_mask]

        if candidate_detect.shape[0] == 0:
            print("No hands found")
            return None, None, None

        # Pick the best bounding box with non maximum suppression
        # the boxes must be moved by the corresponding anchor first
        moved_candidate_detect = candidate_detect.copy()
        moved_candidate_detect[:, :2] = candidate_detect[:, :2] + (candidate_anchors[:, :2] * 256)
        box_ids = non_max_suppression_fast(moved_candidate_detect[:, :4], probabilities)

        # Pick the first detected hand. Could be adapted for multi hand recognition
        box_ids = box_ids[0]

        # bounding box offsets, width and height
        dx,dy,w,h = candidate_detect[box_ids, :4]
        center_wo_offst = candidate_anchors[box_ids,:2] * 256

        # 7 initial keypoints
        keypoints = center_wo_offst + candidate_detect[box_ids,4:].reshape(-1,2)
        side = max(w,h) * self.box_enlarge

        # now we need to move and rotate the detected hand for it to occupy a
        # 256x256 square
        # line from wrist keypoint to middle finger keypoint
        # should point straight up
        # TODO: replace triangle with the bbox directly
        source = self._get_triangle(keypoints[0], keypoints[2], side)
        source -= (keypoints[0] - keypoints[2]) * self.box_shift

        debug_info = {
            "detection_candidates": candidate_detect,
            "anchor_candidates": candidate_anchors,
            "selected_box_id": box_ids,
        }

        return source, keypoints, debug_info

    def preprocess_img(self, img):
        # fit the image into a 256x256 square
        shape = np.r_[img.shape]
        pad = (shape.max() - shape[:2]).astype('uint32') // 2
        img_pad = np.pad(
            img,
            ((pad[0],pad[0]), (pad[1],pad[1]), (0,0)),
            mode='constant')
        img_small = cv2.resize(img_pad, (256, 256))
        img_small = np.ascontiguousarray(img_small)

        img_norm = self._im_normalize(img_small)
        return img_pad, img_norm, pad


    def __call__(self, img):
        img_pad, img_norm, pad = self.preprocess_img(img)

        source, keypoints, _ = self.detect_hand(img_norm)
        if source is None:
            return None, None

        # calculating transformation from img_pad coords
        # to img_landmark coords (cropped hand image)
        scale = max(img.shape) / 256
        Mtr = cv2.getAffineTransform(
            source * scale,
            self._target_triangle
        )

        img_landmark = cv2.warpAffine(
            self._im_normalize(img_pad), Mtr, (256,256)
        )

        joints = self.predict_joints(img_landmark)

        # adding the [0,0,1] row to make the matrix square
        Mtr = self._pad1(Mtr.T).T
        Mtr[2,:2] = 0

        Minv = np.linalg.inv(Mtr)

        # projecting keypoints back into original image coordinate space
        kp_orig = (self._pad1(joints) @ Minv.T)[:,:2]
        box_orig = (self._target_box @ Minv.T)[:,:2]
        kp_orig -= pad[::-1]
        box_orig -= pad[::-1]

        return kp_orig, box_orig
