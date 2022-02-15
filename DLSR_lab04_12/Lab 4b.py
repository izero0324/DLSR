#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
import io
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.onnx
 
# In[2]:


device = torch.device("cpu")
class ResNet(nn.Module):

    def __init__(self, block, layers=[2,2,2,2], width=64, input_size=224, num_classes=2048):
        self.inplanes = width
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, width, layers[0])
        self.layer2 = self._make_layer(block, width*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, width*2*2, layers[2], stride=2)
        self.layer4 = self._make_layer(block, width*2*2*2, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(int(input_size/32), stride=1)
        self.fc = nn.Linear(width*2*2*2*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) 
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1) 
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# In[3]:


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# In[4]:


#model = ResNet(BasicBlock).to(device=device)
model = models.resnet18()
'''
StartTime = time.time()
checkpoint = torch.load('/home/ubuntu/Andrew/lab4_model.pth')
model.load_state_dict(checkpoint())
'''
model.eval()

x = torch.FloatTensor( 512, 3, 244, 244)
print(x.shape)
torch.onnx.export(model, x, "Resnet_4.onnx", export_params = True, opset_version = 10, do_constant_folding=True, input_names=['input'], output_names=['output'])


'''
# In[13]:


print('==> Building model..')

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['skewed_training', 'validation']:
            if phase == 'skewed_training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'skewed_training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'skewed_training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'skewed_training':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[14]:


def main():
    str1 = input('Enter depth(number of layers, default: 2 2 2 2): ')
    layerlist = str1.split()
    layerlist = [int(i) for i in layerlist] 

    str2 = input('Enter width(number of channels, default: 64 ): ')
    width = int(str2)

    str3 = input('Enter resolution(input size, default: 224, the number have to be divisible by 32): ')
    input_size = int(str3)
    
    #To determine if your system supports CUDA
    print("==> Check devices..")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Current device: ",device)

    #Also can print your current GPU id, and the number of GPUs you can use.
    print("Our selected device: ", torch.cuda.current_device())
    print(torch.cuda.device_count(), " GPUs is available")


    print('==> Preparing dataset..')

    data_transforms = {
        'skewed_training': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),    
        'evaluation': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'food11re'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['skewed_training','validation','evaluation']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=2)
                  for x in ['skewed_training','validation','evaluation']}
    global dataset_sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['skewed_training','validation','evaluation']}
    class_names = image_datasets['skewed_training'].classes
    
    model_ft = ResNet(BasicBlock, layerlist, width, input_size, 11)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,
                           num_epochs=25)


    PATH = 'lab1/lab1_model.pht'
    torch.save(model_ft.state_dict(), PATH)


    class_correct = list(0. for i in range(11))
    class_total = list(0. for i in range(11))
    correct_top3 = 0.
    with torch.no_grad():
        for data in dataloaders['evaluation']:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model_ft(images)
            _, predicted = torch.max(outputs, 1)
            _, predicted_top3 = outputs.topk(3, dim=1, largest=True, sorted=True)
            c = (predicted == labels).squeeze()
            c_top3_0 = (predicted_top3[:,0] == labels).squeeze()
            c_top3_1 = (predicted_top3[:,1] == labels).squeeze()
            c_top3_2 = (predicted_top3[:,2] == labels).squeeze()
            if(labels.size()==torch.Size([4])):
                for i in range(4):
                    class_correct[labels[i]] += c[i].item()
                    class_total[labels[i]] += 1
                    correct_top3 += c_top3_0[i].item()
                    correct_top3 += c_top3_1[i].item()
                    correct_top3 += c_top3_2[i].item()
            else:
                for i in range(3):
                    class_correct[labels[i]] += c[i].item()
                    class_total[labels[i]] += 1
                    correct_top3 += c_top3_0[i].item()
                    correct_top3 += c_top3_1[i].item()
                    correct_top3 += c_top3_2[i].item()

    total_correct = 0.
    for i in range(11):
        total_correct+=class_correct[i]

    print('Test set: Top1 Accuracy: %d/%d (%2d %%) , Top3 Accuracy: %d/%d (%2d %%)' % (
        total_correct,dataset_sizes['evaluation'], 100 * total_correct / dataset_sizes['evaluation'],
        correct_top3,dataset_sizes['evaluation'], 100 * correct_top3 / dataset_sizes['evaluation']))
    
    for i in range(11):
        print('class %s : %d/%d %2d %%' % (
            class_names[i],class_correct[i],class_total[i], 100 * class_correct[i] / class_total[i]))


# In[15]:


if __name__ == '__main__':
    main()


# In[ ]:
'''



