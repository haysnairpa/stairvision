import cv2
import time

def display_thread_fn(latest_frames, stop_event):
    fps = 0.0
    last_time = time.time()
    frame_count = 0

    try:
        while not stop_event.is_set():
            any_shown = False
            keys = list(latest_frames.keys())
            for cam_id in keys:
                frame = latest_frames.get(cam_id, None)
                if frame is None:
                    continue
                any_shown = True
                frame_count += 1
                now = time.time()
                if now - last_time >= 1.0:
                    fps = frame_count / (now - last_time)
                    frame_count = 0
                    last_time = now

                disp = frame.copy()
                cv2.putText(disp, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow(f"Stairvision Camera {cam_id}", disp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            if not any_shown:
                time.sleep(0.01)

    except Exception as e:
        print("[Display thread] error:", e)
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[Display thread] stopped")