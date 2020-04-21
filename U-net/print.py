from unet import UNet
import torch
import torch.optim as optim
from train_function import train_model

torch.cuda.set_device(torch.device("cuda", 0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)
model.load_state_dict(torch.load('./save_model'))

params = list(model.named_parameters())

for i in params:
    print(i[0])
    print(i[1].shape)
