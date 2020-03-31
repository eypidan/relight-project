from torch import nn


class FCN(nn.Module):

    def __int__(self, n_class):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 4, 5)
        self.conv2 = nn.Conv2d(4, 16, 5)