from data_provider import load_dataset
from collections import defaultdict
import time
import copy
import torch
import os


def save_train_state(model_dict, best_loss, optimizer, epoch_number):
    torch.save({
        'epoch':epoch_number,
        'model_state_dict': model_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }, "./model_state")


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
    result_data_path = os.path.abspath("../data/mid_train")
    origin_data_path = os.path.abspath("../data/mid_train_dark")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists('./model_state'):
        checkpoint = torch.load('./model_state')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']
        epoch = checkpoint['epoch']
    else:
        best_loss = 1e10
        epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    while epoch < num_epochs:
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()

        # get train and validation dataloader
        dataloaders = load_dataset(origin_data_path, result_data_path, 2)
        # batch size 2, if we have a better gpu, this could be bigger

        # every epoch have two phase, train and validation
        # for train phase

        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])
        epoch_samples = 0
        metrics = defaultdict(float)

        for index, (inputs, results) in enumerate(zip(dataloaders['origin_train'], dataloaders['target_train'])):

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
        metrics = defaultdict(float)
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
        # print("epoch_samples: >>> " + str(epoch_samples))
        if epoch_loss < best_loss:
            print("saving best model")
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            # torch.save(model.state_dict(), "./save_model")

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        save_train_state(best_model_wts, 3000, optimizer, epoch)
        # torch.save(model.state_dict(), "./save_model")
        epoch += 1
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
