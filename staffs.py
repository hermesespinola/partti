import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sliding_window import pyramid, sliding_window
import time
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

gray = cv2.imread(args['image'], cv2.IMREAD_GRAYSCALE)
_, src = cv2.threshold(gray, 0, 100, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
src = cv2.dilate(src, kernel, iterations=5)  # dilate so regions are highlighted

# column sum
x_distribution = np.array(cv2.reduce(src, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0], dtype=float)

# Find max change in x
grad_x = np.gradient(x_distribution)
len_x = len(grad_x)
gradd_x = [x for x in grad_x]
max_left, min_left = np.argmax(grad_x[:len_x//2]), np.argmin(grad_x[:len_x//2])
left_most = min(max_left, min_left)
src[:,:left_most] = grad_x[:left_most] = 0
max_right, min_right = np.argmax(grad_x[len_x//2:][::-1]), np.argmin(grad_x[len_x//2:][::-1])
right_most = len(grad_x) - max(max_right, min_right)
src[:,right_most:] = grad_x[right_most:] = 0

# row sum
y_distribution = cv2.reduce(src, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
y_distribution = np.array(y_distribution).flatten()

# Find max change in y
grad_y = np.gradient(y_distribution)
len_y = len(grad_y)
gradd_y = [y for y in grad_y]
max_left, min_left = np.argmax(grad_y[:len_y//2]), np.argmin(grad_y[:len_y//2])
down_most = min(max_left, min_left)
src[:down_most,:] = grad_y[:down_most] = 0
max_right, min_right = np.argmax(grad_y[len_y//2:][::-1]), np.argmin(grad_y[len_y//2:][::-1])
up_most = len(grad_y) - min(max_right, min_right)
src[up_most:,:] = grad_y[up_most:] = 0

gray = gray[down_most:up_most, left_most:right_most]  # crop
to_crop = cv2.rectangle(src, (left_most, up_most), (right_most, down_most), (0, 255, 255), 3)

_, _tresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
tresh = cv2.dilate(_tresh, kernel, iterations=5)  # dilate to highlight regions

# contours
im, _contours, hierarchy = cv2.findContours(tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = []
staff_crops = []
for con in _contours:
    if cv2.contourArea(con) > 24000:
        cv2.drawContours(tresh, [con], 0, 255, -1)  # contour filling
        contours.append(con)
        epsilon = 0.01 * cv2.arcLength(con, True)
        approx = cv2.approxPolyDP(con, epsilon, True)
        left_most, up_most, right_most, down_most = _tresh.shape[1], -1, -1, _tresh.shape[0]
        for coord in approx:
            cX=coord[0][0]
            cY=coord[0][1]
            if cX < left_most:
                left_most = cX
            if cX > right_most:
                right_most = cX
            if cY < down_most:
                down_most = cY
            if cY > up_most:
                up_most = cY
        staff_crops.append(_tresh[down_most:up_most, left_most:right_most])
valid_blobs = len(contours)
print("valid blobs found: %d" % valid_blobs)

res = cv2.bitwise_and(gray, gray, mask=tresh)
res = cv2.drawContours(res, contours, -1, (255, 0, 0), 2)

columns = 3
plt.subplot2grid((1, columns), (0, 0)), plt.title('original'), plt.imshow(to_crop[..., ::-1])  # BGR -> RGB
plt.subplot2grid((1, columns), (0, 1)), plt.title('cropped & segmented'), plt.imshow(res, cmap='gray', interpolation='nearest')
for i in range(valid_blobs):
    staff_crop = staff_crops[valid_blobs-1-i]

    x_distribution = cv2.reduce(staff_crop, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0]  # column sum

    plt.subplot2grid((valid_blobs, columns), (i, columns-1), rowspan=1), plt.title('notes %d' % (i+1))
    min = staff_crop.shape[1] * 255
    x_1 = -1
    for val in x_distribution[50:len(x_distribution)-50]:
        if val < min:
            min = val
    min += 2000
    # print 'notes %d' % (i+1)
    # print "min: %d" % min
    staff_crop_binary = staff_crop.copy()
    staff_crop = cv2.cvtColor(staff_crop, cv2.COLOR_GRAY2RGB)
    for j in range(len(x_distribution)):
        val = x_distribution[j]
        if val < min and x_1 != -1 and j-x_1 > 5:
            # print "ENDS from %d to %d: %d" % (x_1, j, val)
            longest_consecutive = 0
            count = 0
            y_start = 0
            y_end = staff_crop.shape[0]
            note_horizontal_crop = staff_crop_binary[0:staff_crop.shape[0], x_1:j]
            y_distribution = np.array(cv2.reduce(note_horizontal_crop, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).flatten())  # row sum
            for k in range(len(y_distribution)):
                row_sum = y_distribution[k]
                if row_sum > 0:
                    if count == 0:
                        y_start_temp = k
                    count += 1
                else:
                    if count > longest_consecutive:
                        longest_consecutive = count
                        y_start = y_start_temp
                        y_end = k
                    count = 0
            staff_crop = cv2.rectangle(staff_crop, (x_1, y_start), (j, y_end), (255, 0, 0), 1)
            x_1 = -1
        elif val > min and x_1 == -1:
            # print "STARTS at %d: %d" % (j, val)
            x_1 = j
    plt.imshow(staff_crop)
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
