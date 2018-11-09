import numpy as np
import cv2


def corners(tresh):
    # column sum
    x_distribution = np.array(cv2.reduce(tresh, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0], dtype=float)
    y_pos_horizontal = np.arange(len(x_distribution))

    # row sum
    y_distribution = cv2.reduce(tresh, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    y_distribution = np.array(y_distribution).flatten()
    y_pos_vertical = np.arange(len(y_distribution))

    # Find max change in x
    grad_x = np.gradient(x_distribution)
    len_x = len(grad_x)
    # gradd_x = [x for x in grad_x]
    max_left, min_left = np.argmax(grad_x[:len_x//2]), np.argmin(grad_x[:len_x//2])
    left_most = min(max_left, min_left)
    # grad_x[:left_most] = 0
    max_right, min_right = np.argmax(grad_x[len_x//2:][::-1]), np.argmin(grad_x[len_x//2:][::-1])
    right_most = len(grad_x) - max(max_right, min_right)
    # grad_x[right_most:] = 0

    # Find max change in y
    grad_y = np.gradient(y_distribution)
    len_y = len(grad_y)
    # gradd_y = [y for y in grad_y]
    max_left, min_left = np.argmax(grad_y[:len_y//2]), np.argmin(grad_y[:len_y//2])
    down_most = min(max_left, min_left)
    # grad_y[:down_most] = 0
    max_right, min_right = np.argmax(grad_y[len_y//2:][::-1]), np.argmin(grad_y[len_y//2:][::-1])
    up_most = len(grad_y) - min(max_right, min_right)
    # grad_y[up_most:] = 0

    return (down_most, up_most), (left_most, right_most)
