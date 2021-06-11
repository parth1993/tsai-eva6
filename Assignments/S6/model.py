import torch
import torch.nn as nn
import torch.nn.functional as F

GROUPS = [2, 2, 2, 2, 2, 2, 2] #Groups for group normalization for each conv block
GROUP_COUNTER = 0

def fuse_conv_and_normalization(conv, normalization_type):
    global GROUP_COUNTER
    global GROUPS
    fusedconv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias,
    )
    if normalization_type == "batch":
        nl = nn.BatchNorm2d(conv.out_channels)
    elif normalization_type == "group":
        group = GROUPS[GROUP_COUNTER]
        nl = nn.GroupNorm(group, conv.out_channels)
        GROUP_COUNTER += 1
    elif normalization_type == "layer":
        nl = nn.GroupNorm(1, conv.out_channels)

    return nn.Sequential(fusedconv, nl)


class Net(nn.Module):
    def __init__(self, normalization_type, *args, **kwargs):
        super(Net, self).__init__()
        # RF = 1
        global GROUP_COUNTER
        GROUP_COUNTER = 0

        self.normalization_type = normalization_type
        self.conv1 = nn.Sequential(
            fuse_conv_and_normalization(
                nn.Conv2d(1, 32, 3, padding=1, bias=False), self.normalization_type
            ),
            nn.Dropout(0.01),
            nn.ReLU(),
        )
        # input_size = 28x28x1
        # output_size = 28x28x32
        # RF = 3

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0, bias=False),
            fuse_conv_and_normalization(
                nn.Conv2d(10, 10, 3, padding=1, bias=False), self.normalization_type
            ),
            nn.Dropout(0.01),
            nn.ReLU(),
        )
        # input_size = 28x28x32
        # output_size = 28x28x10
        # RF = 5

        self.pool1 = nn.MaxPool2d(2, 2)
        # input_size = 28x28x10
        # output_size = 14x14x10
        # RF = 10

        self.conv3 = nn.Sequential(
            fuse_conv_and_normalization(
                nn.Conv2d(10, 10, 3, padding=0, bias=False), self.normalization_type
            ),
            nn.Dropout(0.01),
            nn.ReLU(),
        )
        # input_size = 14x14x10
        # output_size = 12x12x10
        # RF = 12

        self.conv4 = nn.Sequential(
            fuse_conv_and_normalization(
                nn.Conv2d(10, 10, 3, padding=0, bias=False), self.normalization_type
            ),
            nn.ReLU(),
        )
        # input_size = 12x12x10
        # output_size = 10x10x10
        # RF = 14

        self.conv5 = nn.Sequential(
            fuse_conv_and_normalization(
                nn.Conv2d(10, 10, 3, padding=0, bias=False), self.normalization_type
            ),
            nn.Dropout(0.01),
            nn.ReLU(),
        )
        # input_size = 10x10x10
        # output_size = 8x8x10
        # RF = 16

        self.conv6 = nn.Sequential(
            fuse_conv_and_normalization(
                nn.Conv2d(10, 16, 3, padding=0, bias=False), self.normalization_type
            ),
            nn.Dropout(0.01),
            nn.ReLU(),
        )
        # input_size = 8x8x10
        # output_size = 6x6x16
        # RF = 18

        self.conv7 = nn.Sequential(
            fuse_conv_and_normalization(
                nn.Conv2d(16, 10, 3, padding=0, bias=False), self.normalization_type
            ),
            nn.Dropout(0.01),
            nn.ReLU(),
        )
        # input_size = 6x6x16
        # output_size = 4x4x10
        # RF = 20

        self.gap1 = nn.AvgPool2d(4)
        # input_size = 4x4x10
        # output_size = 1x1x10
        # RF = 20

        self.fc1 = nn.Sequential(nn.Linear(10, 10, bias=False))
        # input_size = 1x1x10
        # output_size = 1x1x10
        # RF = 20

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap1(x)
        x = x.reshape(-1, 10)
        x = self.fc1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)