#!/usr/local/bin/python3
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import paper_borders_gradient
import staffs_and_notes
import staff_lines
from label_image import predict_from_mat
from pprint import pprint

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

src = cv2.imread(args['image'], cv2.IMREAD_UNCHANGED)
original = src.copy()
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
original = original[vertical_crop[0]:vertical_crop[1], horizontal_crop[0]:horizontal_crop[1]]  # crop
to_crop = cv2.rectangle(src, (horizontal_crop[0], vertical_crop[1]), (horizontal_crop[1], vertical_crop[0]), (0, 255, 255), 3)

# find staffs
contours, staff_crops, original_crops = staffs_and_notes.find_staffs(tresh, _tresh, original)
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
    staff_crop = cv2.cvtColor(staff_crops[valid_blobs-1-blob], cv2.COLOR_GRAY2RGB)
    for note_rect in note_rects[valid_blobs-1-blob]:
        staff_crop = cv2.rectangle(staff_crop, (note_rect[0][0], note_rect[0][1]), (note_rect[1][0], note_rect[1][1]), (255, 0, 0), 1)
    plt.imshow(staff_crop)
plt.show()

heads_rects = [[] for _ in range(len(staff_crops))]
for i in range(len(staff_crops)):
    staff_crop = staff_crops[valid_blobs-1-i]
    # get notes only
    _kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(staff_crop.copy(), cv2.MORPH_OPEN, _kernel)
    opened = cv2.dilate(opened, _kernel, iterations=1)  # dilate to highlight notes
    notes_opened_rects = staffs_and_notes.find_notes([opened], 1)
    opened = cv2.cvtColor(opened, cv2.COLOR_GRAY2RGB)
    for note_rect in notes_opened_rects[0]:
        opened = cv2.rectangle(opened, (note_rect[0][0], note_rect[0][1]), (note_rect[1][0], note_rect[1][1]), (0, 0, 255), 1)
        heads_rects[i].append(((note_rect[0][0], note_rect[0][1]), (note_rect[1][0], note_rect[1][1])))

    # find lines
    lines = staff_lines.find_lines(staff_crop, note_rects[valid_blobs-1-i], False)
    lines_found = len(lines)
    print(lines)

    # find pitch of notes
    for num_note, note_rect in enumerate(notes_opened_rects[0]):
        note_y_center = (note_rect[1][1]+note_rect[0][1]) // 2
        print("note %d center: %d" % (num_note+1, note_y_center))
        min_distance = staff_crop.shape[0]
        closest_line = -1
        for num_line, line in enumerate(list(lines)):
            line_y_center = (line[0][1]+line[1][1]) // 2
            print("\tline %d center: %d" % (num_line+1, line_y_center))
            distance = abs(note_y_center-line_y_center)
            if distance < min_distance:
                min_distance = distance
                closest_line = num_line+1
            # check for intermediate lines
            if num_line < lines_found-1:
                next_line = lines[num_line+1]
                next_line_y_center = (next_line[0][1]+next_line[1][1]) // 2
                intermediate_line_y_center = (line_y_center+next_line_y_center) // 2
                distance = abs(note_y_center - intermediate_line_y_center)
                print("\tline %0.1f center: %d" % (num_line+1 + 0.5, intermediate_line_y_center))
                if distance < min_distance:
                    min_distance = distance
                    closest_line = num_line+1 + 0.5
        print("\tnote %d, closest to line %0.1f" % (num_note, closest_line))
    print("-----")

    cv2.imshow('only notes %d' % (i+1), opened), cv2.waitKey(0), cv2.destroyAllWindows()

def rect_contains(container, rect, offsetX, offsetY):
    x11, y11 = container[0]
    x12, y12 = container[1]
    x21, y21 = rect[0]
    x22, y22 = rect[1]
    return x11 - offsetX <= x21 and x12 + offsetX >= x22 and y11 - offsetY <= y21 and y12 + offsetY >= y22

note_rects_filtered = [[] for _ in range(valid_blobs)]
for i, staff_rects in enumerate(note_rects):
    for note_rect in staff_rects:
        contained = [rect_contains(note_rect, head_rect, 5, 5) for head_rect in heads_rects[i]]
        if any(contained):
            note_rects_filtered[i].append(note_rect)

# filter notes rects with heads
print(note_rects_filtered)
print(heads_rects)
note_crops = [original_crops[i][note_rect[0][1]:note_rect[1][1], note_rect[0][0]-2:note_rect[1][0]+2, :]
    for i, staff_rects in enumerate(note_rects_filtered) for note_rect in staff_rects]

predictions = [predict_from_mat(note) for note in note_crops]
pprint(predictions)
