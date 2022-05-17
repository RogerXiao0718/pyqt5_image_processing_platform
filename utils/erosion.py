import cv2

def erosion(src, erosion_size, morph_shape):

    element = cv2.getStructuringElement(morph_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    erosion_dst = cv2.erode(src, element)
    return erosion_dst