import cv2
import numpy as np

def correct_laser_dot(coord: np.ndarray, img: np.ndarray) -> np.ndarray:
    green = img[:, :, 1]
    x,y = coord

    laser_mask = np.zeros_like(green)
    laser_mask[y-43:y+43, x-43:x+43] = 255

    mask = np.zeros_like(green)
    mask[np.logical_and(245 <= green, laser_mask == 255)] = 255

    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    c = max(contours, key = cv2.contourArea)

    M = cv2.moments(c)
    cX = float(M["m10"] / M["m00"])
    cY = float(M["m01"] / M["m00"])

    return np.array([cX, cY])
