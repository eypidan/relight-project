import os
import imageio
import numpy


def constant_lighten(input):
    input = input.astype(numpy.float32)
    output = input*3
    return output


file_dir = os.walk("../data/example/example_dark")
save_dir = "../data/example/example_Constant"

for path, dir_list, file_list in file_dir:
    print(path)
    for file_name in file_list:
        print("processiong image:%s" % file_name)
        current_path = os.path.join(path, file_name)
        image = imageio.imread(os.path.abspath(current_path))
        img_output = constant_lighten(image)

        imageio.imwrite(os.path.join(save_dir, file_name), img_output)

    print("Done. Save to example_UNet directory.")