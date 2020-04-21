from data_provider import load_dataset
from collections import defaultdict
import time
import copy
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt


def calc_loss(pred, target, metrics):
    loss = torch.abs(target - pred)
    loss = loss.sum() / 64

    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, num_epochs=5):
    result_data_path = os.path.abspath("../data/small_train")
    origin_data_path = os.path.abspath("../data/small_train_dark")


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()

        # get train and validation dataloader
        dataloaders = load_dataset(origin_data_path, result_data_path, 2)

        # every epoch have two phase, train and validation
        # for train phase

        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])
        epoch_samples = 0
        metrics = defaultdict(float)

        for index, (inputs, results) in enumerate(zip(dataloaders['origin_train'], dataloaders['target_train'])):

            # if index % 100 == 0 and index != 0:
            #     torch.save(model.state_dict(), "./save_model")
            # # display test
            # a = inputs[0][0].permute(1, 2, 0)
            # plt.imshow(a)
            # plt.show()
            # b = results[0][0].permute(1, 2, 0)
            # plt.imshow(b)
            # plt.show()
            inputs = inputs[0].to(device)
            results = results[0].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            torch.set_grad_enabled(True)
            outputs = model(inputs)
            loss = calc_loss(outputs, results, metrics)
            loss.backward()
            optimizer.step()

            # statistics
            epoch_samples += inputs.shape[0]
            # if index % 10 == 0:
            #     print(str(index) + "/" + str(len(dataloaders['origin_train'])))
            #     print("loss:" + str(loss))
        scheduler.step()
        print_metrics(metrics, epoch_samples, "train")

        # for validation phase
        epoch_samples = 0
        for index, (inputs, results) in enumerate(zip(dataloaders['origin_validation'], dataloaders['target_validation'])):
            inputs = inputs[0].to(device)
            results = results[0].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            torch.set_grad_enabled(False)
            outputs = model(inputs)
            loss = calc_loss(outputs, results, metrics)
            # statistics
            epoch_samples += inputs.shape[0]

        print_metrics(metrics, epoch_samples, "validation")
        epoch_loss = metrics['loss'] / epoch_samples
        print("epoch_samples: >>> " + str(epoch_samples))
        if epoch_loss < best_loss:
            print("saving best model")
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "./save_model")

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        torch.save(model.state_dict(), "./save_model")

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
