import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt


def pairwise(it):
    it = iter(it)
    while True:
        yield next(it), next(it)


def peaks(y_hist):
    y_hist = enumerate(iter(y_hist))
    _, prev = next(y_hist)
    y, current = next(y_hist)
    _, nxt = next(y_hist)
    if prev < current > nxt:
        yield y, current
    for i, val in y_hist:
        prev = current
        current = nxt
        nxt = val
        if prev < current > nxt:
            yield i, current


def find_bigger_gaps(rects):
    gaps = [(rects[0][1][0], rects[1][0][0]), (rects[1][1][0], rects[2][0][0])]
    gap_sizes = [gaps[0][1] - gaps[0][0], gaps[1][1] - gaps[0][0]]
    for i, rect1 in enumerate(rects):
        if i == len(rects) - 1:
            break
        rect2 = rects[i+1]
        gap = (rect1[1][0], rect2[0][0])
        gap_size = gap[1] - gap[0]
        if gap_size > gap_sizes[0]:
            gaps[1] = gaps[0]
            gap_sizes[1] = gap_sizes[0]
            gaps[0] = gap
            gap_sizes[0] = gap_size
        elif gap_size > gap_sizes[1]:
            gaps[1] = gap
            gap_sizes[1] = gap_size
    return tuple(gaps)


def find_lines(staff, notes_rects, show=False):
    # Find bigger gaps between notes
    gaps = find_bigger_gaps(notes_rects)
    staff_height = staff.shape[1]
    gap_centers = ((gaps[0][1] + gaps[0][0]) // 2, (gaps[1][1] + gaps[1][0]) // 2)

    gap_images = (staff[:, gaps[0][0]:gaps[0][1]], staff[:, gaps[1][0]:gaps[1][1]])
    dist = lambda im: np.array(cv2.reduce(im, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S))
    distributions = (dist(gap_images[0]), dist(gap_images[1]))
    distributions = (distributions[0].flatten(), distributions[1].flatten())

    # Find histogram peaks
    peak_widths = np.arange(1, 3)
    peaks1 = [(i, distributions[0][i]) for i in find_peaks_cwt(distributions[0], peak_widths)]
    peaks2 = [(i, distributions[1][i]) for i in find_peaks_cwt(distributions[1], peak_widths)]
    print('peaks')
    print(peaks1)
    print(peaks2, end='\n')

    # Create five sorted lines with highest points
    p1s = sorted(peaks1, key=lambda x: -x[1])[:5]
    p1s = sorted(p1s, key=lambda x: x[0])
    p1s = map(lambda x: (gap_centers[0], x[0]), p1s)
    p2s = sorted(peaks2, key=lambda x: -x[1])[:5]
    p2s = sorted(p2s, key=lambda x: x[0])
    p2s = map(lambda x: (gap_centers[1], x[0]), p2s)
    lines = list(zip(p2s, p1s))

    # Show results
    print('lines')
    for line in lines:
        print(line)
        cv2.line(staff, line[0], line[1], (255, 0, 0), 3)
    print('')
    if show:
        # plot distributions
        idxs = np.arange(len(distributions[0]))
        plt.bar(idxs, distributions[0].flatten(), align='center')
        plt.bar(idxs, distributions[1].flatten(), align='center')
        plt.show()
        cv2.waitKey()
        cv2.destroyAllWindows()

        cv2.rectangle(staff, (gaps[0][0], 0), (gaps[0][1], staff_height), (255, 0, 0))
        cv2.rectangle(staff, (gaps[1][0], 0), (gaps[1][1], staff_height), (255, 0, 0))
        cv2.imshow('with lines', staff)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return lines
