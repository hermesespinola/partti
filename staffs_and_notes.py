#!/usr/local/bin/python3
import cv2
import numpy as np


def find_staffs(tresh, _tresh):
    im, _contours, hierarchy = cv2.findContours(tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    staff_crops = []
    for con in _contours:
        if cv2.contourArea(con) > 19000:
            cv2.drawContours(tresh, [con], 0, 255, -1)  # contour filling
            contours.append(con)
            x, y, w, h = cv2.boundingRect(con)
            staff_crops.append(_tresh[y:y+h, x:x+w])
    return contours, staff_crops


def find_notes(staff_crops, valid_blobs):
    note_rects = [[] for _ in range(valid_blobs)]
    for i in range(valid_blobs):
        staff_crop = staff_crops[valid_blobs-1-i]

        x_distribution = cv2.reduce(staff_crop, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0]  # column sum

        min = staff_crop.shape[1] * 255
        x_1 = -1
        for val in x_distribution[50:len(x_distribution)-50]:
            if val < min:
                min = val
        min += 1800
        # print 'notes %d' % (i+1)
        # print "min: %d" % min

        for j in range(len(x_distribution)):
            val = x_distribution[j]
            if val < min and x_1 != -1 and j-x_1 > 5:
                # print "ENDS from %d to %d: %d" % (x_1, j, val)
                longest_consecutive = 0
                count = 0
                y_start = 0
                y_end = staff_crop.shape[0]
                note_horizontal_crop = staff_crop[0:staff_crop.shape[0], x_1:j]
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
                note_rects[valid_blobs-1-i].append(((x_1, y_start), (j, y_end)))
                x_1 = -1
            elif val > min and x_1 == -1:
                # print "STARTS at %d: %d" % (j, val)
                x_1 = j
    return note_rects
