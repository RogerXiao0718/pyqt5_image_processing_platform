import cv2

def dilation(src, erosion_size, morph_shape):

    element = cv2.getStructuringElement(morph_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    dilation_dst = cv2.dilate(src, element)
    return dilation_dst