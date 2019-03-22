import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import models
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2

import utils

class VGG(nn.Module):
    def __init__(self, ckpt_path, num_classes):
        super(VGG, self).__init__()
        # vgg_16bn
        model_ft, input_size = utils.initialize_model(model_name='vgg', num_classes=num_classes,
                                                         feature_extract=False,
                                                         use_pretrained=True)

        # ckpt gpu->cpu
        model_ft.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage)['model'])

        self.vgg = model_ft

        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:43]

        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


class DenseNet(nn.Module):
    def __init__(self, ckpt_path, num_classes):
        super(DenseNet, self).__init__()
        model_ft, input_size = utils.initialize_model(model_name='densenet', num_classes=num_classes,
                                                         feature_extract=False,
                                                         use_pretrained=True)

        # ckpt gpu->cpu
        model_ft.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage)['model'])
        self.densenet = model_ft

        # disect the network to access its last convolutional layer
        self.features_conv = self.densenet.features

        # add the average global pool
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        # get the classifier of the vgg19
        self.classifier = self.densenet.classifier

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # don't forget the pooling
        x = self.global_avg_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)

# 还没修改正确
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # define the resnet152
        self.resnet = models.resnet152(pretrained=True)

        # isolate the feature blocks
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                      self.resnet.layer1,
                                      self.resnet.layer2,
                                      self.resnet.layer3,
                                      self.resnet.layer4)

        # average pooling layer
        self.avgpool = self.resnet.avgpool

        # classifier
        self.classifier = self.resnet.fc

        # gradient placeholder
        self.gradient = None

    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def forward(self, x):
        # extract the features
        x = self.features(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # complete the forward pass
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)

        return x