import cv2, time, torch
import numpy as np
from ultralytics import YOLO
from collections import deque

from config import SEG_MODEL_PATH, POSE_MODEL_PATH, RESIZE_WIDTH, RESIZE_HEIGHT, KP_CONF_THRESH
from utils.masks import build_masks
from utils.draw import overlay_mask, draw_pose, draw_fps
from utils.logic import is_on_stair, check_holding

def camera_worker(cam_id, cam_src, queue):
    seg_model = YOLO(SEG_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

    hand_history, status_memory = {}, {}
    cap = cv2.VideoCapture(cam_src)
    if not cap.isOpened():
        print(f"Cannot open camera: {cam_src}")
        return

    frame_num, start_time = 0, time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        h, w = frame.shape[:2]
        frame_vis = frame.copy()

        # Segmentation
        with torch.no_grad():
            try:
                seg_results = seg_model.predict(frame, conf=0.4, verbose=False)
            except: seg_results = []
        handrail_mask, stair_mask = build_masks(seg_results, h, w)

        # Pose
        with torch.no_grad():
            try:
                pose_results = pose_model.predict(frame, conf=0.25, verbose=False)
            except: pose_results = []

        # Mask overlays
        frame_vis = overlay_mask(frame_vis, stair_mask, (255, 0, 0), 0.3)
        frame_vis = overlay_mask(frame_vis, handrail_mask, (255, 255, 0), 0.4)

        # Process people
        for pid, r in enumerate(pose_results):
            if getattr(r, "keypoints", None) is None: continue
            try:
                kpts_xy = r.keypoints.xy.cpu().numpy()
                kpts_conf = r.keypoints.conf.cpu().numpy()
            except: continue

            for person_idx in range(kpts_xy.shape[0]):
                person_kpts, person_conf = kpts_xy[person_idx], kpts_conf[person_idx]
                frame_vis = draw_pose(frame_vis, person_kpts, person_conf)

                if not is_on_stair(person_kpts, person_conf, stair_mask, h, w):
                    continue

                holding_status = False
                for hid in [9, 10]:
                    is_holding, hand_pos = check_holding(
                        person_idx, hid, person_kpts, person_conf,
                        handrail_mask, h, w, hand_history, status_memory
                    )
                    if hand_pos:
                        cv2.circle(frame_vis, hand_pos, 8,
                                   (0, 255, 0) if is_holding else (0, 0, 255), -1)
                    if is_holding: holding_status = True

                label_pos = (int(person_kpts[5][0]), int(person_kpts[5][1]) - 20)
                text = "HOLDING" if holding_status else "NOT HOLDING"
                color = (0, 255, 0) if holding_status else (0, 0, 255)
                cv2.putText(frame_vis, text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # FPS
        frame_num += 1
        fps = frame_num / (time.time() - start_time + 1e-6)
        frame_vis = draw_fps(frame_vis, fps)

        queue.put((cam_id, frame_vis))

    cap.release()
