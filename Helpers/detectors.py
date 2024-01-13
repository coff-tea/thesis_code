"""
Author: Xi Lu
File: detectors.py
Description: Image classification networks from select families created with specified characteristics. Used for dolphin whistle detection (spectrograms).
""" 



#===================================================================================================
import torch
import torch.nn as nn
import torchvision.models as models

from math import floor



#===================================================================================================
class SimpleDetector(nn.Module):
    """
    Simple convolution network used as a detector, with specifications given about number of layers as input

    Args:
        - chs_in (int): Image channels
        - dim_in (int): Image spatial dimension, height=width
        - drop_prob (float): Probability for dropout layers
        - conv_chs (list of int): Channels of convolution layers
        - conv_ks (list of int): Filter sizes of convolution layers
        - nodes (list of int): Hidden layer nodes of dense layers except final
        - classes (int): Number of output classes [1]
    """

    def __init__(self, chs_in, dim_in, drop_prob, conv_chs, conv_ks, nodes, classes=1):
        super(SimpleDetector, self).__init__()
        conv_layers = []
        chs_curr = chs_in
        dim_curr = dim_in
        for i in range(len(conv_chs)):
            conv_layers.append(nn.Conv2d(chs_curr, conv_chs[i], conv_ks[i]))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2))
            chs_curr = conv_chs[i]
            dim_curr = floor((dim_curr-2)/2)
        self.conv = nn.Sequential(*conv_layers)
        self.flat = nn.Flatten()
        dense_layers = []
        dim_curr = dim_curr*dim_curr*chs_curr
        for i in range(len(nodes)):
            dense_layers.append(nn.Linear(dim_curr, nodes[i]))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(drop_prob))
            dim_curr = nodes[i]
        dense_layers.append(nn.Linear(dim_curr, classes))
        self.dense = nn.Sequential(*dense_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.dense(x)
        return x



#===============================================================================================
""" FUNCTION: make_detector
Create and return an image-classification oriented model

Args:
    - model_sel (str): Indicate model to be created
    - chs_in (int): Image channels
    - dim_in (int): Image spatial dimension, height=width
    - freeze (int): Freeze model parameters up to a certain point, model-dependent, only for some models [0, freeze nothing]
    - drop_prob (float): Probability for dropout layers [0.5]
    - rgb_sel (int): Choose one of the RGB channel's pre-trained parameters to replicate [0, choose red]
        - classes (int): Number of output classes [1]
    - pre (bool): Use pretrained parameters rather than random initialisation [True]
"""
def make_detector(model_sel, chs_in, dim_in, freeze=0, drop_prob=0.5, rgb_sel=0, classes=1, pre=True):
    model = None
    
    ####### SIMPLE MODEL
    if model_sel == "simple":
        conv_chs = [16, 32, 64, 64, 64]
        conv_ks = [3, 3, 3, 3, 3]
        nodes = [256, 64, 16]
        model = SimpleDetector(chs_in, dim_in, drop_prob, conv_chs, conv_ks, nodes, classes=classes)
    
    ####### VGG16 REPRODUCE FROM https://arxiv.org/abs/2211.15406
    elif model_sel == "vgg16tf":    # Without drop_prob
        if pre:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
            model = models.vgg16()
        model.classifier = nn.Sequential(
            nn.Linear(25088, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, classes))
    elif model_sel == "vgg16tfd":   # With drop_prob
        if pre:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
            model = models.vgg16()
        model.classifier = nn.Sequential(
            nn.Linear(25088, 50),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(20, classes))
    
    ####### VGG MODELS (16, 16_bn, 19, 19_bn)
    elif "vgg" in model_sel:
        if model_sel == "vgg16":
            if pre:
                model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            else:
                model = models.vgg16()
            fr_layers = [0, 5, 10, 17, 24, -1]
        elif model_sel == "vgg16bn":
            if pre:
                model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
            else:
                model = models.vgg16_bn()
            fr_layers = [0, 7, 14, 24, 34, -1]
        elif model_sel == "vgg19":
            if pre:
                model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            else:
                model = models.vgg19()
            fr_layers = [0, 5, 10, 19, 28, -1]
        else:
            if pre:
                model = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
            else:
                model = models.vgg19_bn()
            fr_layers = [0, 7, 14, 27, 40, -1]
        hold = model.features[0].weight.data[:, rgb_sel:rgb_sel+1, :, :]
        model.features[0] = nn.Conv2d(chs_in, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        for ch in range(chs_in):
            model.features[0].weight.data[:, ch:ch+1, :, :] = hold
        if freeze > 0:
            for i in range(len(fr_layers)):
                if i == len(fr_layers)-1:
                    if i < freeze:
                        for param in model.classifier[0:6].parameters():
                            param.requires_grad = False
                elif i < freeze:
                    start = fr_layers[i]
                    end = fr_layers[i+1]
                    if end == -1:
                        for param in model.features[start:].parameters():
                            param.requires_grad = False
                    else:
                        for param in model.features[start:end].parameters():
                            param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, classes)
    
    ####### RESNET MODELS (50, 101, 152)
    elif "res" in model_sel:
        if model_sel == "res50":
            if pre:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                model = models.resnet50()
        elif model_sel == "res101":
            if pre:
                model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            else:
                model = models.resnet101()
        else:
            if pre:
                model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            else:
                model = models.resnet152()
        hold = model.conv1.weight.data[:, rgb_sel:rgb_sel+1, :, :]
        model.conv1 = nn.Conv2d(chs_in, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        for ch in range(chs_in):
            model.conv1.weight.data[:, ch:ch+1, :, :] = hold
        if freeze > 0:
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.bn1.parameters():
                param.requires_grad = False
        if freeze > 1:
            for param in model.layer1.parameters():
                param.requires_grad = False
        if freeze > 2:
            for param in model.layer2.parameters():
                param.requires_grad = False
        if freeze > 3:
            for param in model.layer3.parameters():
                param.requires_grad = False
        if freeze > 4:
            for param in model.layer4.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(2048, classes, bias=True)
    
    ####### DENSENET MODELS (161, 169, 201)
    elif "dense" in model_sel:
        if model_sel == "dense161":
            if pre:
                model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
            else:
                model = models.densenet161()
            filter_num = 96
            hidden = 2208
        elif model_sel == "dense169":
            if pre:
                model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
            else:
                model = models.densenet169()
            filter_num = 64
            hidden = 1664
        else:
            if pre:
                model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
            else:
                model = models.densenet201()
            filter_num = 64
            hidden = 1920
        hold = model.features.conv0.weight.data[:, rgb_sel:rgb_sel+1, :, :]
        model.features.conv0 = nn.Conv2d(chs_in, filter_num, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        for ch in range(chs_in):
            model.features.conv0.weight.data[:, ch:ch+1, :, :] = hold
        if freeze > 0:
            for param in model.features.conv0.parameters():
                param.requires_grad = False
            for param in model.features.norm0.parameters():
                param.requires_grad = False
        if freeze > 1:
            for param in model.features.denseblock1.parameters():
                param.requires_grad = False
        if freeze > 2:
            for param in model.features.transition1.parameters():
                param.requires_grad = False
        if freeze > 3:
            for param in model.features.denseblock2.parameters():
                param.requires_grad = False
        if freeze > 4:
            for param in model.features.transition2.parameters():
                param.requires_grad = False
        if freeze > 5:
            for param in model.features.denseblock3.parameters():
                param.requires_grad = False
        if freeze > 6:
            for param in model.features.transition3.parameters():
                param.requires_grad = False
        if freeze > 7:
            for param in model.features.denseblock4.parameters():
                param.requires_grad = False
        if freeze > 8:
            for param in model.features.norm5.parameters():
                param.requires_grad = False
        model.classifier = nn.Linear(hidden, classes, bias=True)
    return model