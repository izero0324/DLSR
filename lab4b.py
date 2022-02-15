import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as trans
import torchvision.models as models
import matplotlib.pyplot as plt
# %matplotlib inline
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# from progressbar import *

from imgaug import augmenters as iaa
import PIL
from PIL import Image

import time
import datetime

batch_size = 32
criterion = torch.nn.CrossEntropyLoss()
train_on_gpu = torch.cuda.is_available()
n_epochs = 50
lr = 0.001

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
	          iaa.GammaContrast((0.5, 2.0)),
            # iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

train_datapath = "../lab1/food11re/skewed_training"
valid_datapath = "../lab1/food11re/validation"
test_datapath = "../lab1/food11re/evaluation"

transform＿train = trans.Compose([
    # ImgAugTransform(),
    # lambda x: PIL.Image.fromarray(x),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip(),
    trans.Resize(size=(224,224)),
    trans.ToTensor(),
    trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = trans.Compose([
    trans.Resize(size=(224,224)),
    trans.ToTensor(),
    trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                   
])

data_train = datasets.ImageFolder(train_datapath, transform=transform_train)
data_valid = datasets.ImageFolder(valid_datapath, transform=transform_test)
data_test  = datasets.ImageFolder(test_datapath,  transform=transform_test)


train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader  = torch.utils.data.DataLoader(data_test,  batch_size=batch_size, shuffle=True, num_workers=4)

savepath_ES   = "./model/model4b.pth.tar"
classes = list(i for i in range(11))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(2)


# load pretrained model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 11)

if os.path.isfile(savepath_ES):
    checkpoint = torch.load(savepath_ES)
    model.load_state_dict(checkpoint['model_state_dict'])
    if train_on_gpu:
        model = model.to(device)
        print('training on GPU')
    else:
        print('training on CPU')
    # specify optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum= 0.9)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_to_start = checkpoint['epoch'] + 1
    valid_loss_min = checkpoint['loss']
else:
    epoch_to_start = 1
    if train_on_gpu:
        model = model.to(device)
        print('training on GPU')
    else:
        print('training on CPU')
    # specify optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum= 0.9)
    valid_loss_min = np.Inf # track change in validation loss

print('epoch to start:', epoch_to_start)


# plotting curves
train_loss_plot = []
train_acc_plot = []
valid_loss_plot = []
valid_acc_plot = []

StartTime = time.time()
for epoch in range(epoch_to_start, n_epochs+1):

    correct_train, correct_valid = 0, 0
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    # progress = ProgressBar()
    # for data, target in progress(train_loader):
    for data, target in train_loader:
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        _, pred = output.max(1)
        correct_train += pred.eq(target).sum().item()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        _, pred = output.max(1)
        correct_valid += pred.eq(target).sum().item()
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)

    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
        
    # print training/validation statistics
    valid_acc = 100.*correct_valid/len(data_valid)
    print('Epoch: {} \tTraining Loss: {:.6f}({:.2f}%) \tValidation Loss: {:.6f}({:.4f}%)'.format(
        epoch, train_loss, 100.*correct_train/len(data_train), valid_loss, valid_acc))
    train_loss_plot.append(train_loss)
    train_acc_plot.append(100.*correct_train/len(data_train))
    valid_loss_plot.append(valid_loss)
    valid_acc_plot.append(valid_acc)
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min, valid_loss))
        valid_loss_min = valid_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_loss,
            'acc': valid_acc
            }, savepath_ES)
EndTime = time.time()
print('Time Usage: ', str(datetime.timedelta(seconds=int(round(EndTime-StartTime)))))

# plot training, validation curves
plt.figure()
plt.plot(range(1, n_epochs+1), train_loss_plot, 'b', label='train')
plt.plot(range(1, n_epochs+1), valid_loss_plot, 'g', label='valid')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig("./lab4b_loss.png")
plt.close()

plt.figure()
plt.plot(range(1, n_epochs+1), train_acc_plot, 'b', label='train')
plt.plot(range(1, n_epochs+1), valid_acc_plot, 'g', label='valid')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy(%)')
plt.legend(loc='upper right')
plt.savefig("./lab4b_acc.png")
plt.close()


# track test loss
StartTime = time.time()
checkpoint = torch.load(savepath_ES)
model.load_state_dict(checkpoint['model_state_dict'])

test_loss = 0.0
class_correct = list(0. for i in range(11))
class_total = list(0. for i in range(11))

pred_tot = []
model.eval()
i=1
# iterate over test data
# print(len(test_loader))
for data, target in test_loader:
    i=i+1
    # if len(target)!=batch_size:
    #     continue
        
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)    
    pred_tot.extend(pred.tolist()) 
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(len(target)):       
      label = target.data[i]
      class_correct[label] += correct[i].item()
      class_total[label] += 1
EndTime = time.time()
print('Time Usage: ', str(datetime.timedelta(seconds=int(round(EndTime-StartTime)))))
     
# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(11):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
