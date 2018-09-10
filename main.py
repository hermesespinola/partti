#!/usr/local/bin/python3
import argparse
from document import Scanner
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Path to the image to be scanned")
ap.add_argument("-k", "--ksize", required=False, type=int,
	default=5, help="Size of gaussian kernel to perform blur")
args = vars(ap.parse_args())

scanner = Scanner()
im = cv2.imread(args['image'])
with_contours, transformed = scanner.detect_edge(im, kernel_size=args['ksize'], transform_image=True)

print("Find contours of paper")
cv2.imshow("Outline", with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Wraped image", transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
