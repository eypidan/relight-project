import torch
import os
import matplotlib.pyplot as plt
from unet import UNet
from PIL import Image
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
    # resize image list
    two_exp_list = [2**i for i in range(15)]
    new_height = min(two_exp_list, key=lambda x: x-height if x > height else float("inf"))
    new_weight = min(two_exp_list, key=lambda x: x-weight if x > weight else float("inf"))

    new_input = F.pad(input=input, pad=[0, new_height-height, 0, 0, 0, new_weight-weight], mode='constant', value=0)
    new_input = new_input.unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model_object(input)
    output = output[0].permute(1, 2, 0)
    output = output.cpu().data.numpy()
    plt.imshow(output)
    plt.show()


image = Image.open(os.path.abspath("../data/test/building.png"))

light_enhancement_UNet(image)