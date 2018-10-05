import argparse
import os, glob
import cv2
import numpy as np
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
left = min(max_left, min_left)
src[:,:left] = grad_x[:left] = 0
max_right, min_right = np.argmax(grad_x[len_x//2:][::-1]), np.argmin(grad_x[len_x//2:][::-1])
right = len(grad_x) - max(max_right, min_right)
src[:,right:] = grad_x[right:] = 0

# row sum
y_distribution = cv2.reduce(src, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
y_distribution = np.array(y_distribution).flatten()

# Find max change in y
grad_y = np.gradient(y_distribution)
len_y = len(grad_y)
gradd_y = [y for y in grad_y]
max_left, min_left = np.argmax(grad_y[:len_y//2]), np.argmin(grad_y[:len_y//2])
left = min(max_left, min_left)
src[:left,:] = grad_y[:left] = 0
max_right, min_right = np.argmax(grad_y[len_y//2:][::-1]), np.argmin(grad_y[len_y//2:][::-1])
right = len(grad_y) - min(max_right, min_right)
src[right:,:] = grad_y[right:] = 0

# contours
im, _contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = [con for con in _contours if cv2.contourArea(con) > 15000]
print(len(contours))

plt.subplot(2, 2, 1)
y_pos = np.arange(len(x_distribution))
plt.bar(y_pos, gradd_x, align='center')
plt.title('horizontal')

plt.subplot(2, 2, 3)
y_pos = np.arange(len(y_distribution))
plt.barh(y_pos, gradd_y, align='center')
plt.gca().invert_yaxis()
plt.title('vertical')

plt.subplot(1, 2, 2)
res = cv2.bitwise_and(gray, gray, mask=src)
res = cv2.drawContours(res, contours, -1, (255, 0, 0), 2)
plt.imshow(res, cmap='gray', interpolation='nearest')
plt.title('image')

plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
