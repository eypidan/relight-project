from constantConvert import UNet
import torch
import os

torch.cuda.set_device(torch.device("cuda", 0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)
if os.path.exists('./model_state'):
    checkpoint = torch.load('./model_state')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("stop epoch: " + str(checkpoint['epoch']))
    print("best_loss: "+str(checkpoint['best_loss']))
    for param_group in checkpoint['optimizer_state_dict']['param_groups']:
        print("current LR", param_group['lr'])
else:
    print("not found `model_state` file!")
    exit(1)
