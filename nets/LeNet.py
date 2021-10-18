import sys

import torch
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import ConvTranspose2d
from torch.nn import BatchNorm2d
from torch import flatten


class LeNet(Module):
    def __init__(self, numChannels, label_dim, n_trans_layers):
        super(LeNet, self).__init__()
        # Global parameters
        self.transpose_layers = n_trans_layers
        self.num_input_channels = numChannels
        self.label_dim = label_dim
        self.image_width = label_dim[0]

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=self.num_input_channels, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # # initialize first (and only) set of FC => RELU layers
        # self.fc1 = Linear(in_features=800, out_features=500)
        # self.relu3 = ReLU()
        #
        # # initialize our softmax classifier
        # self.fc2 = Linear(in_features=500, out_features=self.label_dim)
        # self.logSoftmax = LogSoftmax(dim=1)

        # Transpose layers
        if self.transpose_layers == 1:
            self.t_conv1 = ConvTranspose2d(in_channels=self.num_channels, out_channels=1, kernel_size=7, stride=3,
                                           padding=2)
        elif self.transpose_layers == 2:
            self.t_conv1 = ConvTranspose2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=5,
                                           stride=2, padding=1)
            self.bnrorm1 = BatchNorm2d(self.num_channels)
            self.t_conv3 = ConvTranspose2d(in_channels=self.num_channels, out_channels=1, kernel_size=3, stride=2,
                                           padding=1)
        elif self.transpose_layers == 3:
            self.t_conv1 = ConvTranspose2d(in_channels=self.num_channels, out_channels=self.num_channels,
                                           kernel_size=3, padding=1)
            self.bnorm1 = BatchNorm2d(self.num_channels)
            self.t_conv2 = ConvTranspose2d(in_channels=self.num_channels, out_channels=self.num_channels,
                                           kernel_size=3, padding=1)
            self.bnorm2 = BatchNorm2d(self.num_channels)
            self.t_conv3 = ConvTranspose2d(in_channels=self.num_channels, out_channels=1, kernel_size=3, padding=1)
        else:
            print("Introduce a valid number of transpose CNN layers")

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        print(x.shape)
        sys.exit(0)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        # x = flatten(x, 1)
        # x = self.fc1(x)
        # x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        # x = self.fc2(x)
        # output = self.logSoftmax(x)
        # return the output predictions

        if self.cnn_layers == 1:
            x = torch.sigmoid(self.t_conv1(x))
        elif self.cnn_layers == 2:
            x = self.bnorm1(ReLU(self.t_conv1(x)))
            x = torch.sigmoid(self.t_conv3(x))
        elif self.cnn_layers == 3:
            x = self.bnorm1(ReLU(self.conv1(x)))
            x = self.bnorm2(ReLU(self.conv2(x)))
            x = torch.sigmoid(self.conv3(x))

        output = x.view(self.image_width, self.image_width)
        return output
