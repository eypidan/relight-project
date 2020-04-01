from unet import UNet
import torch
import torch.optim as optim
from train_function import train_model

torch.cuda.set_device(torch.device("cuda", 0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet(3).to(device)
optimizer_ft = optim.Adam(model.parameters(), lr=1e-3)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler)