import torch
import torchvision
import os
import matplotlib.pyplot as plt

origin_data = os.path.abspath("../data/train")
dark_data = os.path.abspath("../data/trian_dark")


def load_dataset(data_path):
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((512,512)), torchvision.transforms.ToTensor()])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False,
        pin_memory=True
    )
    return train_loader
#
# a = 5
# for batch_idx, (data, target) in enumerate(load_dataset(origin_data)):
#
#     a = data[0].permute(1, 2, 0)
#     break
#
# for batch_idx, (data, target) in enumerate(load_dataset(dark_data)):
#
#     b = data[0].permute(1, 2, 0)
#     break
#
#
# plt.imshow(a)
# plt.imshow(b)
# plt.show()