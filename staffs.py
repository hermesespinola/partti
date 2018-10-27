import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

src = cv2.imread(args['image'], cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

lower_blue = np.array([80, 50, 50])
upper_blue = np.array([130, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
kernel = np.ones((5, 5), np.uint8)
blue_mask = cv2.dilate(blue_mask, kernel, iterations=2)
blue_mask = cv2.erode(blue_mask, kernel, iterations=2)

im2, blue_contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
left_most, up_most, right_most, down_most = src.shape[1], -1, -1, src.shape[0]
found = 0
print("points of reference:")
for c in blue_contours:
    # calculate moments for each contour
    M = cv2.moments(c)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cArea = cv2.contourArea(c)
    if cArea > 1000:
        found += 1
        if found > 4:
            print("ERROR. More than 4 points of reference.")
            exit(1)
        src = cv2.circle(src, (cX, cY), 15, (0, 255, 0), 3)
        if cX < left_most:
            left_most = cX
        if cX > right_most:
            right_most = cX
        if cY < down_most:
            down_most = cY
        if cY > up_most:
            up_most = cY
    print "\tx:%d  y:%d area:%d" % (cX, cY, cArea)
if found < 4:
    print("ERROR. Not enough points of reference.")
    exit(1)
gray = gray[down_most:up_most, left_most:right_most]  # crop
to_crop = cv2.rectangle(src, (left_most, up_most), (right_most, down_most), (0, 255, 255), 3)

_, _tresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
tresh = cv2.dilate(_tresh, kernel, iterations=5)  # dilate to highlight regions

# contours
im, _contours, hierarchy = cv2.findContours(tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = []
staff_crops = []
# _tresh_2 = _tresh.copy()
for con in _contours:
    if cv2.contourArea(con) > 15000:
        cv2.drawContours(tresh, [con], 0, 255, -1)  # contour filling
        contours.append(con)
        epsilon = 0.01 * cv2.arcLength(con, True)
        approx = cv2.approxPolyDP(con, epsilon, True)
        # cv2.drawContours(_tresh_2, [approx], -1, (255, 255, 255), 3)
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
        # _tresh_2 = cv2.rectangle(_tresh_2, (left_most, up_most), (right_most, down_most), (255, 255, 255), 3)
        staff_crops.append(_tresh[down_most:up_most, left_most:right_most])
valid_blobs = len(contours)
print("valid blobs found: %d" % valid_blobs)

res = cv2.bitwise_and(gray, gray, mask=tresh)
res = cv2.drawContours(res, contours, -1, (255, 0, 0), 2)

columns = 4
plt.subplot2grid((1, columns), (0, 0)), plt.title('original'), plt.imshow(to_crop[..., ::-1])  # BGR -> RGB
plt.subplot2grid((1, columns), (0, 1)), plt.title('cropped & segmented'), plt.imshow(res, cmap='gray', interpolation='nearest')
# plt.subplot2grid((1, columns), (0, 2)), plt.title('binary with approx squares'), plt.imshow(_tresh_2, cmap='gray', interpolation='nearest')
for i in range(valid_blobs):
    plt.subplot2grid((valid_blobs, columns), (i, columns-2), rowspan=1), plt.title('staff %d' % (i+1))
    plt.imshow(staff_crops[valid_blobs-1-i], cmap='gray', interpolation='nearest')

    plt.subplot2grid((valid_blobs, columns), (i, columns-1), rowspan=1), plt.title('distribution %d' % (i+1))
    # column sum
    x_distribution = cv2.reduce(staff_crops[valid_blobs-1-i], 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0]
    y_pos = np.arange(len(x_distribution))
    plt.bar(y_pos, x_distribution, align='center')
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
