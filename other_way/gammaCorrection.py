import cv2
import numpy as np
import os

file_dir = os.walk("../data/example/example_dark")
save_dir = "../data/example/example_gamma"
gamma = 3.0


def gamma_transform(image, gamma = 2.0):
    inv_gamma = 1.0/gamma
    table =np.array([((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


for path, dir_list, file_list in file_dir:
    # print(path)
    for file_name in file_list:
        print("processiong image:%s" % file_name)
        current_path = os.path.join(path, file_name)
        img = cv2.imread(current_path)
        img_output = gamma_transform(img, gamma)
        # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # img_y = img_yuv[:, :, 0]
        #
        # # table about gamma
        # table = np.array(i)
        #
        # img_yuv[:, :, 0] = new_img_y
        #
        # # convert back to rgb
        # img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        cv2.imwrite(os.path.join(save_dir,file_name), img_output)

    print("Done. Save to example_gamma directory.")