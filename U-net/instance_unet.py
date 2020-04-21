import torch
from data_provider import load_dataset
import os
import matplotlib.pyplot as plt
from unet import UNet
import matplotlib.image as mpimg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

model_object = UNet().to(device)
model_object.load_state_dict(torch.load('./save_model'))

input = mpimg.imread(os.path.abspath("../data/test/mydesk.jpg"))
input = torch.from_numpy(input[0:512, 0:512, :]).permute(2, 0, 1)
input = input.unsqueeze(0).float().to(device)
output = model_object(input)
output = output[0].permute(1, 2, 0)
output = output.cpu().data.numpy()
plt.imshow(output)
plt.show()
