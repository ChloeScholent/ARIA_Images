# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 14:05:59 2025

@author: rouxm
"""

# Modèle original du papier des inventeurs de squeezeNet : il faudra sans doutes l'adapter pour le papier sur l'haltéro si ils en ont changé l'architecture

import torch
import torch.nn as nn

class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.expand_activation(torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1))

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=3):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)


class MiniSqueezeNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MiniSqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            #Fire(128, 32, 128, 128),
            #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            #Fire(256, 32, 128, 128),
            #Fire(256, 48, 192, 192),
            #Fire(384, 48, 192, 192),
            #Fire(384, 64, 256, 256),
            #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            #Fire(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)


#un modèle est une classe qui appelle nn.module et est constituée de la fonction __init__ et forward


class selfCNN(nn.Module):
    def __init__(self,num_classes=3): #on parle tjrs d'haltéro
        super(selfCNN,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=5,padding=1,stride=1),  #64 = output_channels arbirtraire kernel_size = un peu arbitraire aussi donc on change#en vrai là j'invente
            nn.ReLu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #output = floor(32.5) = 32
            )
        self.classifier=nn.Sequential(
            )
            
    def forward(self,x):   #j'ai copié collé et pas encore bien compris
    
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)
