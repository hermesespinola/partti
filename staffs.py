import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

src = cv2.imread(args['image'], cv2.IMREAD_GRAYSCALE)
cv2.bitwise_not(src, src)  # invert grayscale
kernel = np.ones((5, 5), np.uint8)
src = cv2.dilate(src, kernel, iterations=2)  # dilate so regions are highlighted

# column sum
x_distribution = cv2.reduce(src, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0]
# row sum
y_distribution = cv2.reduce(src, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
y_distribution = np.array(y_distribution).flatten()

plt.subplot(2, 2, 1)
y_pos = np.arange(len(x_distribution))
plt.bar(y_pos, x_distribution, align='center')
plt.title('horizontal')

plt.subplot(2, 2, 3)
y_pos = np.arange(len(y_distribution))
plt.barh(y_pos, y_distribution, align='center')
plt.gca().invert_yaxis()
plt.title('vertical')

plt.subplot(1, 2, 2)
plt.imshow(src, cmap='gray', interpolation='nearest')
plt.title('image')

plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
