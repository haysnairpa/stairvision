import cv2
import numpy as np
from config import CLOSE_KERNEL_SIZE, DILATE_KERNEL_SIZE, ERODE_KERNEL_SIZE

def build_masks(seg_results, h, w):
    """Build handrail & stair masks from segmentation results."""
    handrail_mask = np.zeros((h, w), dtype=np.uint8)
    stair_mask = np.zeros((h, w), dtype=np.uint8)

    for r in seg_results:
        if getattr(r, "masks", None) is not None:
            try:
                for mask_poly, cls in zip(r.masks.xy, r.boxes.cls):
                    poly = np.array(mask_poly, dtype=np.int32)
                    if int(cls) == 0:  # handrail
                        cv2.fillPoly(handrail_mask, [poly], 255)
                    elif int(cls) == 1:  # stair
                        cv2.fillPoly(stair_mask, [poly], 255)
            except Exception:
                continue

    # Morphology filters
    close_kernel = np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8)
    dilate_kernel = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)
    erode_kernel = np.ones((ERODE_KERNEL_SIZE, ERODE_KERNEL_SIZE), np.uint8)

    handrail_mask = cv2.morphologyEx(handrail_mask, cv2.MORPH_CLOSE, close_kernel)
    handrail_mask = cv2.dilate(handrail_mask, dilate_kernel, iterations=1)
    handrail_mask = cv2.erode(handrail_mask, erode_kernel, iterations=1)

    stair_mask = cv2.morphologyEx(stair_mask, cv2.MORPH_CLOSE, close_kernel)
    stair_mask = cv2.dilate(stair_mask, dilate_kernel, iterations=1)
    stair_mask = cv2.erode(stair_mask, erode_kernel, iterations=1)

    return handrail_mask, stair_mask
