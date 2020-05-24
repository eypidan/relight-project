import torch
import os
from unet import UNet
import imageio
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy
import matplotlib.pyplot as plt #debug

def deal_with_small_pitch(input):
    # b = data[0].permute(1, 2, 0)

    # plt.imshow(input)
    # plt.show() #debug

    # device choice
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    # load U-net model
    model_object = UNet().to(device)
    if os.path.exists('./model_state'):
        checkpoint = torch.load('./model_state')
        model_object.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("not found `model_state` file!")
        exit(1)

    # read image and resize image to 2 exponential times
    input = TF.to_tensor(numpy.array(input).copy())
    # input = TF.to_tensor(input)
    height = input.shape[1]
    weight = input.shape[2]
    # resize image
    two_exp_list = [2 ** i for i in range(15)]
    new_height = min(two_exp_list, key=lambda x: x - height if x > height else float("inf"))
    new_weight = min(two_exp_list, key=lambda x: x - weight if x > weight else float("inf"))

    new_input = F.pad(input=input, pad=[0, new_weight - weight, 0, new_height - height, 0, 0], mode='constant', value=0)

    ttt = new_input.permute(1, 2, 0)
    # plt.imshow(ttt)
    # plt.show()# debug

    new_input = new_input.unsqueeze(0).float().to(device)
    # enhancement
    with torch.no_grad():
        output = model_object(new_input)
    # resize back and save
    output = output[0].permute(1, 2, 0)
    output = output.cpu().data.numpy()

    # plt.imshow(output)
    # plt.show()  # debug

    output = output[0:height, 0:weight, :]

    return output


def light_enhancement_UNet(input):
    height = input.shape[0]
    weight = input.shape[1]
    if height <= 2048 and weight <= 2048:
        output = deal_with_small_pitch(input)
        return output
    else:  # big image need to be divided
        # Define the window size
        # count = 0
        windowsize_r = 1024
        windowsize_c = 1024
        output = input.astype(numpy.float32)
        # Crop out the window and calculate the histogram
        for r in range(0, input.shape[0], windowsize_r):
            for c in range(0, input.shape[1], windowsize_c):

                if c + windowsize_c >= weight:
                    c_size = weight - c
                else:
                    c_size = windowsize_c
                if r + windowsize_r >= height:
                    r_size = height - r
                else:
                    r_size = windowsize_r
                window = input[r:r + r_size, c:c + c_size]
                output_window = deal_with_small_pitch(window)
                output[r:r + r_size, c:c + c_size] = output_window
                # file_name_debug = str(count) + "aaa_.jpg"
                # if count == 0:
                #     imageio.imwrite(os.path.join(save_dir, file_name_debug), input[r:r + r_size, c:c + c_size])
                # count = count + 1
        return output


file_dir = os.walk("../data/example/example_dark_noise")
save_dir = "../data/example/example_UNet"

for path, dir_list, file_list in file_dir:
    print(path)
    for file_name in file_list:
        print("processiong image:%s" % file_name)
        current_path = os.path.join(path, file_name)
        image = imageio.imread(os.path.abspath(current_path))

        img_output = light_enhancement_UNet(image)

        imageio.imwrite(os.path.join(save_dir, file_name), img_output)

    print("Done. Save to example_UNet directory.")
