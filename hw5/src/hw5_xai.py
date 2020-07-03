"""hw5_XAI"""

"""Import Packages"""
import os
import cv2
import sys
import numpy as np
from PIL import Image, ImageFilter, ImageChops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.models as models
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace

"""Specify Path"""
model_fname  = 'model_best.model'
data_set_dir = sys.argv[1]
output_dir   = sys.argv[2]

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

# load model
model =  torch.load(model_fname)

"""Dataset"""
class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):
        # mode: 'train' or 'eval'
        
        self.paths = paths
        self.labels = labels
        # data augmentation when training
        train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(), # flip horizontally
            transforms.RandomRotation(15), # rotate
            transforms.ColorJitter(brightness=0.3, saturation=0.3, contrast=0.3),
            transforms.ToTensor(), # turn image into Tensor, and normalize to [0,1](data normalization)
        ])

        # don't need data augmentation when testing
        eval_transform = transforms.Compose([
            transforms.ToPILImage(),                                    
            transforms.ToTensor(),
        ])

        self.transform = trainTransform if mode == 'train' else eval_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = cv2.imread(self.paths[index])
        X = cv2.resize(X,(128, 128))
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)

# parse in data path, return images' path and class
def get_paths_labels(path):
    imgnames = os.listdir(path)
    imgnames.sort()
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels

train_paths, train_labels = get_paths_labels(os.path.join(data_set_dir, 'training'))
# when initializing dataset, only parse path and class
# dataset's __getitem__ method will load image dynamically
train_set = FoodDataset(train_paths, train_labels, mode='eval')

"""Saliency Map"""
def normalize(image):
	# nomalize image
    return (image - image.min()) / (image.max() - image.min())

def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.cuda()

    # input image also needs to calculate gradient
    x.requires_grad_()
    
    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliencies = x.grad.abs().detach().cpu()
    # saliencies: (batches, channels, height, weight)
    # normalize each image because gradient of each image may varies greatly
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies

print('Saliency map')
# specify images' indices that want to visualize
img_indices = [83, 4218, 4707, 8598]
images, labels = train_set.getbatch(img_indices)
saliencies = compute_saliency_maps(images, labels, model)

# transform from cv2(bgr) to pil(rgb)
for idx in range(len(images)):
    image = images[idx]
    b, g, r = image[0].clone(), image[1].clone(), image[2].clone()
    image[0], image[1], image[2] = r, g, b
    images[idx] = image
    save_image(image, os.path.join(output_dir, 'orig_{}.png'.format(idx)))
    save_image(saliencies[idx], os.path.join(output_dir, 'sal_{}.png'.format(idx)))

"""Filter Explaination"""
def normalize(image):
	# normalize image
  	return (image - image.min()) / (image.max() - image.min())

layer_activations = None
def filter_explaination(x, model, cnnid, filterid, iteration=100, lr=1):
	# x: which part of image activates the specified filter
	# cnnid, filterid: which cnn layer and which filter
	model.eval()
	
	def hook(model, input, output):
	  	global layer_activations
	  	layer_activations = output
	
	# record the activation map of certain filter in certain layer (pre-define, not forward() yet)
	hook_handle = model.cnn[cnnid].register_forward_hook(hook)
	
	# Filter activation: observe the activation map when x pass through the specified filter
	model(x.cuda())
	filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
	
	# Filter visualization: find image that activate the filter the most
	x = x.cuda()
	x.requires_grad_()
	optimizer = Adam([x], lr=lr)
	
	# enlarge filter activation with optimizer and gradient
	for iter in range(iteration):
	    optimizer.zero_grad()
	    model(x)
	    objective = -layer_activations[:, filterid, :, :].sum()  
		# compute gradient 
	    objective.backward()
	    # modified image
	    optimizer.step()
	    
	filter_visualization = x.detach().cpu().squeeze()[0]
	
	# important to remove hook
	hook_handle.remove()
	
	return filter_activations, filter_visualization

print('Filter explaination')
img_indices = [83, 4218, 4707, 8598]
images, labels = train_set.getbatch(img_indices)
filter_activations, filter_visualization = filter_explaination(images, model, cnnid=15, filterid=0, iteration=100, lr=0.1)

# transform from cv2(bgr) to pil(rgb)
for idx in range(len(images)):
    image = images[idx]
    b, g, r = image[0].clone(), image[1].clone(), image[2].clone()
    image[0], image[1], image[2] = r, g, b
    images[idx] = image

save_image(normalize(filter_visualization), os.path.join(output_dir, 'filter_visualization.png'))
for idx in range(len(filter_activations)):
    save_image(normalize(filter_activations[idx]), os.path.join(output_dir, 'filter_activation_{}.png'.format(idx)))

"""Lime"""
def predict(input):
    # input: numpy array, (batches, height, width, channels)    
    model.eval()                                                                                                                                                             
    # turn input into pytorch tensor, and transform to (batches, channels, height, width)
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)                                                                                                            

    output = model(input.cuda())                                                                                                                                             
    return output.detach().cpu().numpy()                                                                                                                              
                                                                                                                                                                             
def segmentation(input):
	# slice image into 100 slices
    return slic(input, n_segments=100, compactness=1, sigma=1)                                                                                                                                                                                                                                                                                      

print('Lime')
img_indices = [83, 4218, 4707, 8598]
images, labels = train_set.getbatch(img_indices)

np.random.seed(16)                                                                                                                                                       
for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):                                                                                                                                             
    # lime takes numpy array
    x = image.astype(np.double)

    # two main steps
    explainer = lime_image.LimeImageExplainer()                                                                                                                              
    explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)

    lime_img, mask = explaination.get_image_and_mask(label=label.item(),                                                                                                                           
					                                 positive_only=False,                                                                                                                         
					                                 hide_rest=False,                                                                                                                             
					                                 num_features=11,                                                                                                                              
					                                 min_weight=0.05)
    # transform from cv2(bgr) to pil(rgb)
    b, g, r = lime_img[:, :, 0].copy(), lime_img[:, :, 1].copy(), lime_img[:, :, 2].copy()
    lime_img[:, :, 0], lime_img[:, :, 1], lime_img[:, :, 2] = r, g, b
    
    im = Image.fromarray(np.uint8(lime_img * 255))
    im.save(os.path.join(output_dir, 'lime_img_{}.png'.format(idx)))

"""Deep Dream"""
model = torch.load(model_fname)
model = model.cuda()
model.eval()

# Class to register a hook on the target layer (used to get the output channels of the layer)
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()
  
# Function to make gradients calculations from the output channels of the target layer  
def get_gradients(net_in, net, layer):     
    net_in = net_in.unsqueeze(0).cuda()
    net_in.requires_grad = True
    net.zero_grad()
    hook = Hook(layer)
    net_out = net(net_in)
    loss = hook.output[0].norm()
    loss.backward()
    return net_in.grad.data.squeeze()

# Function to run the dream.
def dream(image_tensor, net, layer, iterations, lr):
    image_tensor = image_tensor.cuda()
    for i in range(iterations):
        gradients = get_gradients(image_tensor, net, layer)
        image_tensor.data = image_tensor.data + lr * gradients.data

    img_out = image_tensor.detach().cpu()
    img_out_np = img_out.numpy().transpose(1,2,0)
    img_out_np = np.clip(img_out_np, 0, 1)

    b, g, r = np.copy(img_out_np[:, :, 0]), np.copy(img_out_np[:, :, 1]), np.copy(img_out_np[:, :, 2])
    img_out_np[:, :, 0], img_out_np[:, :, 1], img_out_np[:, :, 2] = r, g, b

    img_out_pil = Image.fromarray(np.uint8(img_out_np * 255))
    return img_out_pil

# Input image
img_indices = [83, 4218, 4707, 8598]
imgs, labels = train_set.getbatch(img_indices)
layer = model.cnn[15]

idx = 0
for img in imgs:
    img = dream(img, model, layer, 20, 0.4)
    # img = Image.fromarray(np.uint8(img * 255))
    img.save(os.path.join(output_dir, 'dream_img_{}.png'.format(idx)))

    idx = idx + 1