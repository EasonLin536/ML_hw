"""hw9_unsupervised"""

"""Import Packages"""
import os
import sys
import glob
import random
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans

"""Path Specification"""
data_fname  = sys.argv[1]
model_fname = sys.argv[2]

"""Prepare Training Data"""
def preprocess(image_list):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (9000, 32, 32, 3)
    Returns:
      image_list: List of images (9000, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list

class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images

"""Utils"""
def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

"""Model"""
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # input [3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1), # [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2), # [64, 16, 16]
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2), # [128, 8, 8]
            nn.Conv2d(128, 256, 3, stride=1, padding=1), # [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2), # [256, 4, 4]
        )
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x  = self.decoder(x1)
        return x1, x

"""Training"""
def training(img_dataloader, model, n_epoch):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # training
    model.train()
    for epoch in range(n_epoch):
        # adjust_learning_rate(optimizer, epoch, optimizer.param_groups[0]['lr'])
        
        for data in img_dataloader:
            img = data
            img = img.cuda()

            output1, output = model(img)
            loss = criterion(output, img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
        print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, loss.data))

    # save model
    torch.save(model.state_dict(), model_fname)

"""Main"""
# preprocess data
trainX = np.load(data_fname)
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)

model = AE().cuda()
n_epoch = 100
same_seeds(0)
img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)
training(img_dataloader, model, n_epoch)