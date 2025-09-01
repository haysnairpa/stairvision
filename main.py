import multiprocessing as mp
import threading
from workers.camera import camera_worker
from utils.display import display_thread_fn

def main():
    mp.set_start_method("spawn")
    queue = mp.Queue(maxsize=2)
    manager = mp.Manager()
    latest_frames = manager.dict()

    cameras = {
        0: r"D:\Aldi\stairvision\src\dataset\west\videos\Copy of IMG_4568.MOV",
        1: r"D:\Aldi\stairvision\src\dataset\west\videos\Copy of IMG_4567.MOV",
        2: r"D:\Aldi\stairvision\src\dataset\west\videos\Copy of Copy of IMG_3101.MOV",
        3: r"D:\Aldi\stairvision\src\dataset\west\videos\Copy of Copy of IMG_3095.MOV",
    }

    procs = []
    for cam_id, cam_src in cameras.items():
        p = mp.Process(target=camera_worker, args=(cam_id, cam_src, queue), daemon=True)
        p.start()
        procs.append(p)
        latest_frames[cam_id] = None

    display_stop = threading.Event()
    disp_thread = threading.Thread(target=display_thread_fn, args=(latest_frames, display_stop), daemon=True)
    disp_thread.start()

    try:
        while not display_stop.is_set():
            try:
                cam_id, frame_vis = queue.get(timeout=1.0)
                latest_frames[cam_id] = frame_vis
            except Exception:
                pass

    except KeyboardInterrupt:
        print("[Main] KeyboardInterrupt received, shutting down...")
    finally:
        display_stop.set()
        for p in procs:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        try:
            disp_thread.join(timeout=2.0)
        except Exception:
            pass
        print("[Main] exit complete")

if __name__ == "__main__":
    main()