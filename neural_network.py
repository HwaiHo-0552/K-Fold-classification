#########################################################################
#                       训练网络的构建、设定                            #
#                    SAIL LAB, 机器学习研发                             #
#                           2021.5.26                                   #
#########################################################################
#!/usr/bin/python
#-*- coding:UTF-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

#########################################################################
#                           Transfer Learning                           #
#########################################################################
class ResNet:
    def __init__(self, class4recognition):
        self.class4recognition = class4recognition

    def module(self):
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        
        # Here the size of each output sample is set to 2.
        # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, self.class4recognition)

        return model_ft