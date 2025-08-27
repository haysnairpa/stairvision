import cv2, multiprocessing as mp
from workers.camera import camera_worker

def main():
    mp.set_start_method("spawn")
    manager = mp.Manager()
    queue = manager.Queue()

    cameras = {
        0: r"D:\Aldi\stairvision\src\dataset\west\videos\Copy of IMG_4568.MOV",
        1: r"D:\Aldi\stairvision\src\dataset\west\videos\Copy of IMG_4567.MOV",
        2: r"D:\Aldi\stairvision\src\dataset\west\videos\Copy of Copy of IMG_3095.MOV",
        3: r"D:\Aldi\stairvision\src\dataset\west\videos\Copy of IMG_4565.MOV",
    }

    procs = []
    for cam_id, cam_src in cameras.items():
        p = mp.Process(target=camera_worker, args=(cam_id, cam_src, queue))
        p.start()
        procs.append(p)

    try:
        while True:
            try:
                cam_id, frame_vis = queue.get(timeout=2)
                cv2.imshow(f"Stairvision Camera {cam_id}", frame_vis)
            except: pass
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        for p in procs: p.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
