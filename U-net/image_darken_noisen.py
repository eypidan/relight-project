import os
import imageio
import numpy as np
from scipy.ndimage.filters import gaussian_filter

file_dir = os.walk("../data/mid_train/1")
save_dir = "../data/mid_train_dark_noise/1"


# dark the image
def transform(img, a, b, r):
    # a * I
    outImage = (img*float(a))
    outImage[outImage>255] = 255 # bigger than 255, change to 255
    # gamma
    outImage = np.power(outImage/255.0, r) * 255

    # b *
    outImage = (outImage*float(b))
    return outImage


# noise the image
def noise(img):

    PEAK = 50
    # poisson noise
    noisyPoisson = np.random.poisson(img * 255.0 / PEAK) * PEAK / 255

    # gauss noise
    row, col, ch = img.shape
    mean = 0
    var = 1.5
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisyGauss = img + gauss

    # speckle noise
    speckle=np.random.randn(row, col, ch)
    noisySpeckle = img + img*speckle / 10.0

    noisy = noisyGauss/4.0 + noisyPoisson/2.0 + noisySpeckle/4.0
    noisy[noisy > 255] = 255
    noisy[noisy < 0] = 0
    return noisy


# blur the image
def blur(img):
    kernel_size = (5, 5)
    sigma = 0.7
    blur = gaussian_filter(img,sigma)
    return blur

for path, dir_list, file_list in file_dir:
    print(path)
    for file_name in file_list:
        print("processing image:%s" % file_name)
        current_path = os.path.join(path, file_name)
        image = imageio.imread(os.path.abspath(current_path))

        a = np.random.uniform(0.9, 1)
        b = np.random.uniform(0.5, 1)
        r = np.random.uniform(1.5, 5)
        darkImage = transform(image, a, b, r)
        noiseImage = noise(darkImage)
        # blurImage = blur(noiseImage)
        imageio.imwrite(os.path.join(save_dir, file_name), noiseImage)

    print("Done. Save to " + save_dir)