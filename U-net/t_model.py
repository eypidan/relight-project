import torch
from data_provider import load_dataset
import os
import matplotlib.pyplot as plt
from unet import UNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

model_object = UNet().to(device)
model_object.load_state_dict(torch.load('./save_model'))

result_data_path = os.path.abspath("../data/example/example_origin")
origin_data_path = os.path.abspath("../data/example/example_dark")

dataloaders = load_dataset(origin_data_path, result_data_path, 1)

for index, (inputs, results) in enumerate(zip(dataloaders['origin_train'], dataloaders['target_train'])):

    # display test
    a = inputs[0][0].permute(1, 2, 0)
    a.to(cpu_device)
    plt.imshow(a)
    plt.show()

    b = model_object(inputs[0].to(device))
    b = b[0].permute(1, 2, 0)
    b = b.cpu().data.numpy()
    plt.imshow(b)
    plt.show()
    break