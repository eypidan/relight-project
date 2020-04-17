import numpy as np
import cv2 as cv


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


# Gamma Correction
def adjust_gamma(image, gamma=2.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)


def traditional_light_enhancement(low_img):
    # White balance
    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    wb_img = wb.balanceWhite(low_img)
    # Denoise
    dn_img = cv.fastNlMeansDenoisingColored(wb_img,None,10,10,7,15)
    # sharpen
    sp_img = unsharp_mask(dn_img)
    # gamma_correction
    gm_img = adjust_gamma(sp_img, 3.0)
    return gm_img



