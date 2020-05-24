from traditional import traditional_light_enhancement
import cv2
import os

file_dir = os.walk("../data/evaluation/dark_noise")
save_dir = "../data/evaluation/Traditional"


for path, dir_list, file_list in file_dir:
    print(path)
    for file_name in file_list:
        print("processiong image:%s" % file_name)
        current_path = os.path.join(path, file_name)
        img = cv2.imread(current_path)
        img_output = traditional_light_enhancement(img)

        cv2.imwrite(os.path.join(save_dir,file_name), img_output)

    print("Done. Save to example_traditional directory.")