import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # RF = 1
        b1 = 16
        b2 = 30
        b3 = 50
        b4 = 100
        b5 = 256
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, b1, 3, padding=1, bias=False),
            nn.BatchNorm2d(b1),
            nn.Dropout(0.01),
            nn.ReLU(),

            nn.Conv2d(b1, b2, 3, padding=1, bias=False),
            nn.BatchNorm2d(b2),
            nn.Dropout(0.01),
            nn.ReLU(),

            nn.Conv2d(b2, b3, 3, padding=1, bias=False),
            nn.BatchNorm2d(b3),
            nn.Dropout(0.01),
            nn.ReLU()
        ) 
        # input_size = 32x32x3
        # output_size = 32x32x64
        # RF = 7

        self.trans1 = nn.Sequential(
            
            nn.Conv2d(b3, b3, 2, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(b3),
            nn.Dropout(0.01),
            nn.ReLU()
            
        ) 
        # input_size = 32x32x64
        # output_size = 15x15x64
        # RF = 14
        

        self.conv2 = nn.Sequential(
            nn.Conv2d(b3, b3, kernel_size=3, padding=1, groups=b3),
            #pointwise
            nn.Conv2d(b3, b3, kernel_size=1),
            nn.BatchNorm2d(b3),
            nn.Dropout(0.01),
            nn.ReLU(),
        
            #depthwise
            nn.Conv2d(b3, b3, kernel_size=3, padding=1, groups=b3),
            #pointwise
            nn.Conv2d(b3, b4, kernel_size=1),
            nn.BatchNorm2d(b4),
            nn.Dropout(0.01),
            nn.ReLU(),
            
            
            #depthwise
            nn.Conv2d(b4, b4, kernel_size=3, padding=1, groups=b4),
            #pointwise
            nn.Conv2d(b4, b3, kernel_size=1),
            nn.BatchNorm2d(b3),
            nn.Dropout(0.01),
            nn.ReLU(),
        )

        # input_size = 15x15x64
        # output_size = 15x15x64
        # RF = 18

        self.trans2 = nn.Sequential(
            nn.Conv2d(b3, b3, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(b3),
            nn.Dropout(0.01),
            nn.ReLU()
        )
        # input_size = 15x15x64
        # output_size = 7x7x64
        # RF = 36

        self.conv3 = nn.Sequential(
            nn.Conv2d(b3, b1, 1, padding=0, bias=False),
            nn.Conv2d(b1, b3, 3, padding=1, bias=False),
            nn.BatchNorm2d(b3),
            nn.Dropout(0.01),
            nn.ReLU(),

            nn.Conv2d(b3, b2, kernel_size=1, padding=0),
            nn.Conv2d(b2, b3, kernel_size=3, padding=1),
            nn.BatchNorm2d(b3),
            nn.Dropout(0.01),
            nn.ReLU(),

            # depthwise
            nn.Conv2d(b3, b3, kernel_size=3, padding=1, groups=b3),
            # pointwise
            nn.Conv2d(b3, b2, kernel_size=1),
            nn.BatchNorm2d(b2),
            nn.Dropout(0.01),
            nn.ReLU(),
        )
        # input_size = 7x7x64
        # output_size = 7x7x64
        # RF = 40
        
        self.trans3 = nn.Sequential(
            nn.Conv2d(b2, b2, 3, padding=1, dilation=2, bias=False),
            nn.BatchNorm2d(b2),
            nn.Dropout(0.01),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(b2, 10, 1, padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AvgPool2d(6)            
            
        )

        # self.gap1 = nn.AvgPool2d(10) 
        # input_size = 4x4x10
        # output_size = 1x1x10
        # RF = 20

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.trans2(x)
        x = self.conv3(x)
        x = self.trans3(x)
        x = self.conv4(x)
        x = x.reshape(-1, 10)
        return F.log_softmax(x, dim=-1)