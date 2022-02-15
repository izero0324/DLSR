import torchvision.models as models
from thop import profile
import torch
import torch.nn as nn
from thop import clever_format
from my_resnet import *

input = torch.randn(1, 3, 224, 224)

d = 1
w = 1
r = 1

model = ResNet_modify(d,w,r)
print(model)

macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(d, w, r)
print('Total params : ', params)
print('Total MACs : ', macs, '\n')