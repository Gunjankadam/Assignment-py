import cv2
import matplotlib.pyplot as plt
img1 = cv2.imread('lambo_png.jpg')
img2 = cv2.imread('tree.jpg')

img3 = cv2.resize(img1, (560, 560))
img4 = cv2.resize(img2, (560, 560))

img = cv2.hconcat([img3, img4])
plt.imshow(img)
cv2.imshow('merge',img)
cv2.waitKey(0)
cv2.destroyAllWindows()