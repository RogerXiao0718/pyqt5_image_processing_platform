import cv2
import numpy as np

def perspectiveTransform(src, perspectiveTransform_pst_src, perspectiveTransform_pts_dst):
    perspectiveTransform_pts_dst = np.array(perspectiveTransform_pts_dst, dtype=np.float32)
    opt_mapping_matrix, status = cv2.findHomography(perspectiveTransform_pst_src,
                                                    perspectiveTransform_pts_dst)
    transformed_image = cv2.warpPerspective(src, opt_mapping_matrix, (src.shape[1], src.shape[0]))
    return transformed_image