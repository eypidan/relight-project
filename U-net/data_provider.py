import torch
import torchvision
# import os
# import matplotlib.pyplot as plt


def load_dataset(origin_data_path, target_data_path, batch_size=4):
    # origin dataset
    train_dataset = torchvision.datasets.ImageFolder(
        root=origin_data_path,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize((512, 512)), torchvision.transforms.ToTensor()])
    )

    train_set, validation_set = torch.utils.data.random_split(train_dataset, [20000, len(train_dataset) - 20000])
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True
    )

    # target dataset
    target_dataset = torchvision.datasets.ImageFolder(
        root=target_data_path,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize((512, 512)), torchvision.transforms.ToTensor()])
    )
    target_train_set = torch.utils.data.Subset(target_dataset, train_set.indices)
    target_validation_set = torch.utils.data.Subset(target_dataset, validation_set.indices)

    target_train_loader = torch.utils.data.DataLoader(
        target_train_set,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True
    )

    target_validation_loader = torch.utils.data.DataLoader(
        target_validation_set,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True
    )

    data_loader = {
        "origin_train": train_loader,
        "origin_validation": validation_loader,
        "target_train": target_train_loader,
        "target_validation": target_validation_loader
    }

    return data_loader


# # test code
# result_data_path = os.path.abspath("../data/train")
# origin_data_path = os.path.abspath("../data/trian_dark")
#
# dataloader = load_dataset(origin_data_path, result_data_path)
#
# for batch_idx, (data, target) in enumerate(dataloader["origin_train"]):
#
#     a = data[0].permute(1, 2, 0)
#     break
#
# for batch_idx, (data, target) in enumerate(dataloader["target_train"]):
#
#     b = data[0].permute(1, 2, 0)
#     break
#
#
# plt.imshow(a)
# plt.show()
# plt.imshow(b)
# plt.show()
