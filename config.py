SEG_MODEL_PATH = "../src/model/best_stair_handrail_model.pt"
POSE_MODEL_PATH = "../src/model/best_pose_model.pt"

RESIZE_WIDTH, RESIZE_HEIGHT = 640, 360

KP_CONF_THRESH = 0.25
HAND_RADIUS = 20
HAND_OVERLAP_RATIO = 0.2
SMOOTH_FRAMES = 3

DILATE_KERNEL_SIZE = 4
CLOSE_KERNEL_SIZE = 4
ERODE_KERNEL_SIZE = 3

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]