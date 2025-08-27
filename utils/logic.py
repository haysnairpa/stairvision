import cv2
import numpy as np
from collections import deque
from config import KP_CONF_THRESH, HAND_RADIUS, HAND_OVERLAP_RATIO, SMOOTH_FRAMES

def is_on_stair(person_kpts, person_conf, stair_mask, h, w):
    """Check if person is on stair based on hips & ankles."""
    for sid in [11, 12, 15, 16]:  
        if person_kpts.shape[0] > sid and person_conf[sid] > KP_CONF_THRESH:
            sx, sy = int(person_kpts[sid][0]), int(person_kpts[sid][1])
            if 0 <= sx < w and 0 <= sy < h and stair_mask[sy, sx] > 0:
                return True
    return False

def check_holding(person_idx, hid, person_kpts, person_conf, handrail_mask, h, w,
                  hand_history, status_memory):
    """Check if a hand is holding the handrail."""
    if person_kpts.shape[0] <= hid or person_conf[hid] < KP_CONF_THRESH:
        return False, None

    hx, hy = person_kpts[hid]
    hand_history.setdefault((person_idx, hid), deque(maxlen=SMOOTH_FRAMES))
    hand_history[(person_idx, hid)].append((hx, hy))
    avg_hx = int(np.mean([p[0] for p in hand_history[(person_idx, hid)]]))
    avg_hy = int(np.mean([p[1] for p in hand_history[(person_idx, hid)]]))

    if not (0 <= avg_hx < w and 0 <= avg_hy < h):
        return False, (avg_hx, avg_hy)

    y_grid, x_grid = np.ogrid[:h, :w]
    mask_circle = (x_grid - avg_hx) ** 2 + (y_grid - avg_hy) ** 2 <= HAND_RADIUS ** 2
    inside_mask = np.logical_and(mask_circle, handrail_mask > 0)
    overlap_ratio = inside_mask.sum() / (mask_circle.sum() + 1e-6)

    status_memory.setdefault((person_idx, hid), deque(maxlen=SMOOTH_FRAMES))
    status_memory[(person_idx, hid)].append(overlap_ratio >= HAND_OVERLAP_RATIO)
    is_holding = sum(status_memory[(person_idx, hid)]) >= (SMOOTH_FRAMES // 2 + 1)

    return is_holding, (avg_hx, avg_hy)
