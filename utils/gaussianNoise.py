import cv2
import numpy as np

def gaussianNoise(src, mean=0, sigma=0.1):
    normalized_image = src / 255
    noise = np.random.normal(mean, sigma, normalized_image.shape)

    gaussian_out = normalized_image + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)

    gaussian_out = np.uint8(gaussian_out * 255)
    return gaussian_out