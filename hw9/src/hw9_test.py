"""hw9_unsupervised"""

"""Import Packages"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans

"""Path Specification"""
data_fname    = sys.argv[1]
model_fname   = sys.argv[2]
predict_fname = sys.argv[3]

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

"""Fix Random Seed"""
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

"""Testing"""
def inference(X, model, batch_size=256):
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    print('Latents Shape:', latents.shape)
    return latents

def predict(latents):
    # First Dimension Reduction
    transformer = KernelPCA(n_components=400, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    # # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

def invert(pred):
    return np.abs(1-pred)

def save_prediction(pred, out_csv='prediction.csv'):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')

"""Main"""
# testing
same_seeds(0)
print('Loading model: {}'.format(model_fname))
model = AE().cuda()
model.load_state_dict(torch.load(model_fname))
model.eval()

trainX = np.load(data_fname)

# predict
latents = inference(X=trainX, model=model)
pred, X_embedded = predict(latents)

# save to csv
# save_prediction(pred, predict_fname)
save_prediction(invert(pred), predict_fname)