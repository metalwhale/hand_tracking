import cv2
from src.hand_tracker import HandTracker
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"
MULTIHAND = True
HULL = True
POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
HULL_COLOR = (0, 0, 255)
THICKNESS = 1
HULL_THICKNESS = 2


def drawpointstoframe(points, frame):
    if points is not None:
        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, HULL_THICKNESS)
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
        if HULL:
            for hull_connection in hull_connections:
                x0, y0 = points[hull_connection[0]]
                x1, y1 = points[hull_connection[1]]
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), HULL_COLOR, HULL_THICKNESS)



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

    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 9), (0, 13)
]
pseudo_hull_connections = [(0, 17), (17, 18), (18, 19), (19, 20), (0, 1), (1, 2), (2, 3), (3, 4)]
hull_connections = [(4, 8), (8, 12), (12, 16), (16, 20)]
if HULL:
    hull_connections += pseudo_hull_connections
else:
    connections += pseudo_hull_connections
detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)




while hasFrame:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points1, points2, _, _ = detector(image)
    drawpointstoframe(points1, frame)
    if MULTIHAND:
        drawpointstoframe(points2, frame)
    cv2.imshow(WINDOW, frame)
    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
