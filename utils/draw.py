import cv2
import numpy as np
from config import SKELETON_CONNECTIONS, KP_CONF_THRESH

def draw_pose(frame_vis, person_kpts, person_conf):
    """Draw pose skeleton on frame."""
    for i in range(person_kpts.shape[0]):
        if person_conf[i] > KP_CONF_THRESH:
            x, y = int(person_kpts[i][0]), int(person_kpts[i][1])
            cv2.circle(frame_vis, (x, y), 3, (200, 200, 200), -1)

    for s, e in SKELETON_CONNECTIONS:
        if person_kpts.shape[0] > max(s, e):
            if person_conf[s] > KP_CONF_THRESH and person_conf[e] > KP_CONF_THRESH:
                cv2.line(frame_vis,
                         tuple(np.array(person_kpts[s], int)),
                         tuple(np.array(person_kpts[e], int)),
                         (255, 255, 255), 2)
    return frame_vis

def overlay_mask(frame, mask, color, alpha=0.4):
    """Overlay a binary mask on top of frame."""
    if mask is None or mask.sum() == 0:
        return frame
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame
