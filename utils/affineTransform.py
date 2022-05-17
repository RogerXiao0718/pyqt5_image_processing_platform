import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

def affineTransform(src, rotateValue, translateXValue, translateYValue, flip):
    H = src.shape[0]
    W = src.shape[1]
    center = (int(W / 2), int(H / 2))
    degree = rotateValue * 2 / math.pi
    rotationMatrix = cv2.getRotationMatrix2D(center, degree, scale=1)
    transformed_image = cv2.warpAffine(src, rotationMatrix, (W, H))
    translateMatrix = np.float32([
        [1, 0, translateXValue],
        [0, 1, translateYValue]
    ])
    transformed_image = cv2.warpAffine(transformed_image, translateMatrix, (W, H))
    if flip:
        transformed_image = cv2.flip(transformed_image, flipCode=1)
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    plt.imshow(transformed_image)
    plt.axis("off")
    plt.show()