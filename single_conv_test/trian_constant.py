from constantConvert import Constant
import torch
import torch.optim as optim
from train_function import train_model

torch.cuda.set_device(torch.device("cuda", 0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = Constant().to(device)
optimizer_ft = optim.Adam(model.parameters(), lr=1e-3)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.9)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=80)
