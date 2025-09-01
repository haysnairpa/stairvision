import cv2
import numpy as np
from ultralytics import YOLO
import torch
import traceback
from config import SEG_MODEL_PATH, POSE_MODEL_PATH, RESIZE_WIDTH, RESIZE_HEIGHT, SKELETON_CONNECTIONS
from utils.masks import build_masks
from utils.logic import is_on_stair, check_holding

def camera_worker(cam_id, cam_src, queue):
    try:
        seg_model = YOLO(SEG_MODEL_PATH)
        pose_model = YOLO(POSE_MODEL_PATH)

        hand_history = {}
        status_memory = {}

        cap = cv2.VideoCapture(cam_src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[Worker {cam_id}] ❌ Cannot open source: {cam_src}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESIZE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 15)

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            N = 2
            if frame_idx % N != 0:
                continue

            frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            h, w = frame.shape[:2]

            with torch.no_grad():
                try:
                    seg_results = seg_model.predict(frame, conf=0.4, verbose=False)
                except Exception as e:
                    print(f"[Worker {cam_id}] Segmentation error:", e)
                    seg_results = []

            handrail_mask, stair_mask = build_masks(seg_results, h, w)

            with torch.no_grad():
                try:
                    pose_results = pose_model.predict(frame, conf=0.25, verbose=False)
                except Exception as e:
                    print(f"[Worker {cam_id}] Pose error:", e)
                    pose_results = []

            frame_vis = frame.copy()

            if stair_mask is not None and stair_mask.sum() > 0:
                stair_overlay = frame_vis.copy()
                stair_overlay[stair_mask > 0] = (255, 0, 0)
                frame_vis = cv2.addWeighted(stair_overlay, 0.25, frame_vis, 0.75, 0)

            if handrail_mask is not None and handrail_mask.sum() > 0:
                hr_overlay = frame_vis.copy()
                hr_overlay[handrail_mask > 0] = (255, 255, 0)
                frame_vis = cv2.addWeighted(hr_overlay, 0.3, frame_vis, 0.7, 0)

            for pid, r in enumerate(pose_results):
                if getattr(r, "keypoints", None) is None:
                    continue
                try:
                    kpts_xy = r.keypoints.xy.cpu().numpy()
                    kpts_conf = r.keypoints.conf.cpu().numpy()
                except Exception:
                    continue

                for person_idx in range(kpts_xy.shape[0]):
                    person_kpts = kpts_xy[person_idx]
                    person_conf = kpts_conf[person_idx]

                    for i in range(person_kpts.shape[0]):
                        if person_conf[i] > 0.25:
                            x, y = int(person_kpts[i][0]), int(person_kpts[i][1])
                            cv2.circle(frame_vis, (x, y), 3, (200, 200, 200), -1)

                    for s, e in SKELETON_CONNECTIONS:
                        if person_kpts.shape[0] > max(s, e):
                            if person_conf[s] > 0.25 and person_conf[e] > 0.25:
                                cv2.line(frame_vis,
                                         tuple(np.array(person_kpts[s], int)),
                                         tuple(np.array(person_kpts[e], int)),
                                         (255, 255, 255), 2)

                    on_stair = is_on_stair(person_kpts, person_conf, stair_mask, h, w)
                    if not on_stair:
                        continue

                    holding_status_for_person = False
                    for hid in [9, 10]:
                        is_holding, avg_hand = check_holding(
                            person_idx, hid, person_kpts, person_conf,
                            handrail_mask, h, w, hand_history, status_memory
                        )
                        if avg_hand is not None:
                            cv2.circle(frame_vis, avg_hand, 8,
                                       (0, 255, 0) if is_holding else (0, 0, 255), -1)
                        if is_holding:
                            holding_status_for_person = True

                    label_pos = (int(person_kpts[5][0]), int(person_kpts[5][1]) - 20) if person_kpts.shape[0] > 5 else (10, 30)
                    text = "HOLDING" if holding_status_for_person else "NOT HOLDING"
                    color = (0, 255, 0) if holding_status_for_person else (0, 0, 255)
                    cv2.putText(frame_vis, text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            try:
                if not queue.full():
                    queue.put((cam_id, frame_vis), block=False)
                else:
                    try:
                        _ = queue.get(block=False)
                        queue.put((cam_id, frame_vis), block=False)
                    except Exception:
                        pass
            except Exception:
                break

        cap.release()

    except Exception as ex:
        print(f"[Worker {cam_id}] Unexpected error:", ex)
        traceback.print_exc()
    finally:
        try:
            cap.release()
        except Exception:
            pass
        print(f"[Worker {cam_id}] terminated")