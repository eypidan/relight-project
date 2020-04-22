import torch
import os
from unet import UNet
import imageio
import torchvision.transforms.functional as TF
import torch.nn.functional as F


def light_enhancement_UNet(input):
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

    input = TF.to_tensor(input)
    height = input.shape[1]
    weight = input.shape[2]
    # resize image
    two_exp_list = [2**i for i in range(15)]
    new_height = min(two_exp_list, key=lambda x: x-height if x > height else float("inf"))
    new_weight = min(two_exp_list, key=lambda x: x-weight if x > weight else float("inf"))

    new_input = F.pad(input=input, pad=[0, new_weight-weight, 0, new_height - height, 0, 0], mode='constant', value=0)
    new_input = new_input.unsqueeze(0).float().to(device)
    # enhancement
    with torch.no_grad():
        output = model_object(new_input)
    # resize back and save
    output = output[0].permute(1, 2, 0)
    output = output.cpu().data.numpy()
    output = output[0:height, 0:weight, :]
    return output


file_dir = os.walk("../data/example/example_dark")
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