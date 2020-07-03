"""hw3_CNN"""

"""Import Packages"""
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from   torch.utils.data import DataLoader, Dataset
import time

"""Specify Path"""
image_dir = sys.argv[1]
model_dir = sys.argv[2]

"""Read Image"""
def readfile(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x

print("Reading data")
train_x, train_y = readfile(f"{image_dir}/training",   True)
val_x, val_y     = readfile(f"{image_dir}/validation", True)
print(f"Size of training data = {len(train_x)}")
print(f"Size of validation data = {len(val_x)}")

"""Dataset"""
# data augmentation when training
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), # flip horizontally
    transforms.RandomRotation(15), # rotate
    transforms.ColorJitter(brightness=0.3, saturation=0.3, contrast=0.3),
    transforms.ToTensor(), # turn image into Tensor, and normalize to [0,1](data normalization)
])

# don't need data augmentation when testing
test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

batch_size   = 128
train_set    = ImgDataset(train_x, train_y, train_transform)
val_set      = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)

"""Model"""
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input dimension [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.RReLU(0.001,0.005),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.RReLU(0.001,0.005),
            nn.MaxPool2d(2, 2, 0), # [64, 64, 64]
            nn.Dropout(0.3),
 
            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.RReLU(0.001,0.005),
            nn.MaxPool2d(2, 2, 0), # [128, 32, 32]
            nn.Dropout(0.3),
 
            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.RReLU(0.001,0.005),
            nn.MaxPool2d(2, 2, 0), # [256, 16, 16]
            nn.Dropout(0.3),
 
            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.RReLU(0.001,0.005),
            nn.MaxPool2d(2, 2, 0), # [512, 8, 8]
            nn.Dropout(0.3),
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.RReLU(0.001,0.005),
            nn.MaxPool2d(2, 2, 0), # [512, 4, 4]
            nn.Dropout(0.3),
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 11)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

"""Training with train & val set"""
model = Classifier().cuda()
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nstart training, parameter total:{total}, trainable:{trainable}\n')

loss = nn.CrossEntropyLoss() # classification task -> CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer: Adam
num_epoch = 150

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc, train_loss, val_acc, val_loss = 0.0, 0.0, 0.0, 0.0

    model.train() # model: train model (use Dropout...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() #  initialize gradients of model with optimizer
        train_pred = model(data[0].cuda()) # get prediction dustribution with model(model.forward)
        batch_loss = loss(train_pred, data[1].cuda()) # compute loss(prediction and label should be in the same device)
        
        batch_loss.backward() # compute gradient with back propagation
        optimizer.step() # update parameters with optimizer and gradient

        train_acc  += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred   = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc  += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # print the results
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()), end=' ')


"""Train with all data"""
train_val_x      = np.concatenate((train_x, val_x), axis=0)
train_val_y      = np.concatenate((train_y, val_y), axis=0)
train_val_set    = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

model_best = Classifier().cuda()
loss = nn.CrossEntropyLoss() # classification task -> CrossEntropyLoss
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001) # optimizer: Adam
num_epoch = 150

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc  = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        
        batch_loss.backward()
        optimizer.step()

        train_acc  += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        # print the results
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))

torch.save(model_best, model_dir)