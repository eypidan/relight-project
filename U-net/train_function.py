from data_provider import load_dataset
import time
import copy
import torch
import torch.nn.functional as F

def calc_loss(pred, target):
    loss = torch.abs(target - pred)
    loss = loss.sum() / 64
    return loss


def train_model(model, optimizer, scheduler, num_epochs=2):
    origin_data = "../data/train"
    result_data = "../data/train_dark"

    dataloaders = {
        'train_origin': load_dataset(origin_data),
        'train_result': load_dataset(result_data),
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_set, validation_set = torch.utils.data.random_split(dataloaders['train_origin'])

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()
        model.trian()

        for inputs, results in zip(dataloaders['train_origin'], dataloaders['train_result']):
            inputs = inputs.to(device)
            results = results.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = calc_loss(outputs, results)
            loss.backward()
            optimizer.step()
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


