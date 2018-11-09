#!/usr/local/bin/python3
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import paper_borders_gradient
import staffs_and_notes
import staff_lines

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

src = cv2.imread(args['image'], cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# binarization
_, _tresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
tresh = cv2.dilate(_tresh, kernel, iterations=5)  # dilate to highlight regions

# crop with gradients
vertical_crop, horizontal_crop = paper_borders_gradient.corners(tresh)
gray = gray[vertical_crop[0]:vertical_crop[1], horizontal_crop[0]:horizontal_crop[1]]  # crop
tresh = tresh[vertical_crop[0]:vertical_crop[1], horizontal_crop[0]:horizontal_crop[1]]  # crop
_tresh = _tresh[vertical_crop[0]:vertical_crop[1], horizontal_crop[0]:horizontal_crop[1]]  # crop
to_crop = cv2.rectangle(src, (horizontal_crop[0], vertical_crop[1]), (horizontal_crop[1], vertical_crop[0]), (0, 255, 255), 3)

## I'll keep this, just in case we need it
# _kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# closed = cv2.morphologyEx(_tresh, cv2.MORPH_CLOSE, _kernel)
# opened = cv2.morphologyEx(_tresh, cv2.MORPH_OPEN, _kernel)
## Finding lines
# without_notes = _tresh - opened

# find staffs
contours, staff_crops = staffs_and_notes.find_staffs(tresh, _tresh)
valid_blobs = len(staff_crops)
print("valid blobs found: %d" % valid_blobs)

# mask staffs with contours
res = cv2.bitwise_and(gray, gray, mask=tresh)
res = cv2.drawContours(res, contours, -1, (255, 0, 0), 2)

# find notes
note_rects = staffs_and_notes.find_notes(staff_crops, valid_blobs)

# show results
display_columns = 3
plt.subplot2grid((1, display_columns), (0, 0)), plt.title('original'), plt.imshow(to_crop[..., ::-1])  # BGR -> RGB
plt.subplot2grid((1, display_columns), (0, 1)), plt.title('cropped & segmented'), plt.imshow(res, cmap='gray', interpolation='nearest')
for blob in range(valid_blobs):
    plt.subplot2grid((valid_blobs, display_columns), (blob, display_columns-1), rowspan=1), plt.title('notes %d' % (blob+1))
    staff_crop = cv2.cvtColor(staff_crops[blob], cv2.COLOR_GRAY2RGB)
    for note_rect in note_rects[blob]:
        staff_crop = cv2.rectangle(staff_crop, (note_rect[0][0], note_rect[0][1]), (note_rect[1][0], note_rect[1][1]), (255, 0, 0), 1)
    plt.imshow(staff_crop)
plt.show()

for i, staff_crop in enumerate(staff_crops):
    staff_lines.find_lines(staff_crop, note_rects[valid_blobs-1-i], True)
