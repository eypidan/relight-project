import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def convert(img):
    b,g,r = cv.split(img)
    s_img = cv.merge([r,g,b])
    return s_img

low_img = cv.imread("low.jpg")
normal = cv.imread("normal.jpg")
plt.subplot(231);plt.imshow(convert(low_img))
plt.subplot(232);plt.imshow(convert(normal))

# White balance
wb = cv.xphoto.createGrayworldWB()
wb.setSaturationThreshold(0.99)
wb_img = wb.balanceWhite(convert(low_img))
plt.subplot(233);plt.imshow(wb_img)

# Denoise
dn_img = cv.fastNlMeansDenoisingColored(wb_img,None,10,10,7,15)
plt.subplot(234);plt.imshow(dn_img)

# Sharpen
def unsharp_mask(image, kernel_size = (5,5), sigma=1.0,amount=1.0,threshold=0):
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

sp_img = unsharp_mask(dn_img)
plt.subplot(235);plt.imshow(sp_img)

# Gamma Correction
def adjust_gamma(image,gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)

gm_img = adjust_gamma(sp_img,2.0)
plt.subplot(236);plt.imshow(gm_img)

plt.show()
