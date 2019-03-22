import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.kthvalue as topk

import utils

os.environ['CUDA_VISIBLE_DIVICE'] = '2'
# --------------------------------------------------------------------------------
# ---- Class to generate heatmaps (CAM)

class HeatmapGenerator():

    def __init__(self, pathModel, model_name, num_classes, device='cuda'):

        # ---- Initialize the network
        model_ft, input_size = utils.initialize_model(model_name, num_classes, feature_extract=False,
                                                   use_pretrained=True)

        device = torch.device(device if torch.cuda.is_available() else "cpu")
        model_ft = model_ft.to(device)

        model_ft.load_state_dict(torch.load(pathModel)['model'])


        self.transCrop = input_size
        self.model = model_ft.features
        self.model.eval()

        # ---- Initialize the weights

        self.weights = list(self.model.parameters())[-2]
        # print(len(list(self.model.parameters())[-1]))
        #
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad == True:
        #         print("\t", name)
        # print(list(self.model.parameters()))

        # for name, param in self.model.named_parameters():
        #     print(name, '      ', param.size())
        print('=------------------------------------------------------------------------s')
        print(self.model)


        # ---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(self.transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)

        self.transformSequence = transforms.Compose(transformList)

    # --------------------------------------------------------------------------------

    def generate(self, pathImageFile, pathOutputFile):

        # ---- Load image, transform, convert
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)

        input = torch.autograd.Variable(imageData)

        self.model.cuda()
        output = self.model(input.cuda())
        print(np.shape(output))

        # ---- Generate heatmap
        heatmap = None
        print(len(self.weights))
        print(self.weights)
        for i in range(0, len(self.weights)):
            map = output[0, i, :, :]
            if i == 0:
                heatmap = self.weights[i] * map
            else:
                heatmap += self.weights[i] * map

        # ---- Blend original and heatmap
        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (self.transCrop, self.transCrop))

        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (self.transCrop, self.transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        img = heatmap * 0.5 + imgOriginal

        cv2.imwrite(pathOutputFile, img)

    def getCAM(self, feature_conv, weight_fc, class_idx):
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return [cam_img]


# --------------------------------------------------------------------------------

if __name__ == '__main__':
    model_name = 'densenet'
    num_classes = 4

    pathInputImage = 'test/c6_img20170407_11031488.jpg'
    pathOutputImage = 'heatmapImages/heatmap.png'
    pathModel = '/home/njuciairs/Hejia/local_LogAndCkpt/ckpt/densenet_26_ckpt.pkl'


    h = HeatmapGenerator(pathModel, model_name, num_classes)
    h.generate(pathInputImage, pathOutputImage)
