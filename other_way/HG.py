import cv2
import numpy as np
import os

file_dir = os.walk("../data/evaluation/dark_noise")
save_dir = "../data/evaluation/HG"
for path,dir_list,file_list in file_dir:
    # print(path)
    for file_name in file_list:
        current_path = os.path.join(path,file_name)
        img = cv2.imread(current_path)
        #convert tp yuv color
        img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # convert back to rgb
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        cv2.imwrite(os.path.join(save_dir,file_name), img_output)

    print("Done.Save to example_HG directory.")