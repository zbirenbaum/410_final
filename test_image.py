import cv2
import numpy as np
def remove_white_borders(image):
    image = cv2.bitwise_not(image)
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return cv2.bitwise_not(image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)])

img = cv2.imread('./blues00000.png')
cv2.imshow('noborder', remove_white_borders(img))
cv2.waitKey(0)
