import cv2

def createMorphWindow(window_name, trackBar_element_name, trackBar_kernel_size_name, onChange):
    cv2.namedWindow(window_name)
    cv2.createTrackbar(trackBar_element_name, window_name, 0, 2, onChange)
    cv2.createTrackbar(trackBar_kernel_size_name, window_name, 0, 21, onChange)