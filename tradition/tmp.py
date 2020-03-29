import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("low_light.jpg")
low = img[:,:1605,:]
normal = img[:,1605:,:]
cv.imwrite('low.jpg',low)
cv.imwrite('normal.jpg',normal)
# plt.subplot(121);plt.imshow(low)
# plt.subplot(122);plt.imshow(normal)
# plt.show()
