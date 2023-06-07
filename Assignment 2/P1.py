import cv2
from cv2.gapi import kernel
import numpy as np

img= cv2.imread("dog.jpg")
kernel = np.ones((5, 5), np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)

img = cv2.resize(img, (560, 560))

img_square = img.copy()
cv2.rectangle(img_square,
              pt1=(100,50), #top left
              pt2=(200,150), #bottom right
              color=(255,0,0), #BGR
              thickness=10);
cv2.imshow('rectangle', img_square)
cv2.imshow('img1', img1)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()