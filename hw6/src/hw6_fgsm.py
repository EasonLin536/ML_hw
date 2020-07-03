"""hw6_Adversarial_Attack (iterative fgsm)"""

"""Import Packages"""
import os
import sys
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

device = torch.device("cuda")

"""Specify Path"""
data_dir       = sys.argv[1]
input_img_dir  = f'{data_dir}/images'
label_fname    = f'{data_dir}/labels.csv'
output_img_dir = sys.argv[2]

"""Dataset"""
# inherit torch.utils.data.Dataset to read in image
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        # data path
        self.root = root
        # parse in label from main function
        self.label = torch.from_numpy(label).long()
        # Attacker's transforms transform image into forms matching the model
        self.transforms = transforms
        # list of image file names
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        # read in image with path
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        # transform images
        img = self.transforms(img)
        # label the images
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        # there are total 200 images
        return 200

"""Model : FGSM attack"""
class Attacker:
    def __init__(self, img_dir, label):
        # pre-trained model:
        self.model = models.densenet121(pretrained = True)
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # normalize image to 0~1 mean 0 variance 1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        # use Adverdataset to read data
        self.dataset = Adverdataset(input_img_dir, label, transform)
        
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)

    # FGSM attack
    def i_fgsm_attack(self, x, y, epsilon, iteration, threshold):
        x_adv = Variable(x.data, requires_grad=True)
        success, fail, max_i = 0, 0, 0
        
        for i in range(iteration):
            h_adv = self.model(x_adv)
            loss = F.nll_loss(h_adv, y)
            
            self.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            loss.backward()
            data_grad = x_adv.grad.data

            # find direction of gradient
            sign_data_grad = data_grad.sign()
            # add noise of gradient*epsilon to the image
            x_adv = x_adv + epsilon * sign_data_grad
            
            # set modification threshold
            x_adv = torch.where(x_adv > x+threshold, x+threshold, x_adv)
            x_adv = torch.where(x_adv < x-threshold, x-threshold, x_adv)

            # define new leaf variable
            x_adv = Variable(x_adv.data, requires_grad=True)
            
            # parse image with noise into the model and get corresponding class
            h_adv = self.model(x_adv)
            final_pred = h_adv.max(1, keepdim=True)[1]
                    
            if final_pred.item() == y.item():
                # if iteration is over, then failed
                if i == iteration - 1: fail = 1
            else:
                # if it predicts wrongly, then succeed
                max_i, success = i, 1
                break

        return x_adv, success, fail, max_i
    
    def attack(self, epsilon, iteration):
        # save images that attacked successfully
        adv_result = []
        wrong, fail, success = 0, 0, 0    
        
        # information for tuning eps and iteration
        max_iter = 0
        num_of_zeros = 0
        
        idx = 0
        for data, target in self.loader:
            if idx % 40 == 0: print('processing image', idx)
            print('processing image', idx, end='\r')
            data, target = data.to(device), target.to(device)

            # parse in image to the model to get the corresponding class
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            # if class is wrong, then don't attack
            if init_pred.item() != target.item():
                wrong += 1
            
            else:
                # if class is correct, start computing gradient for FGSM attack
                x = Variable(data, requires_grad=True)
                y = Variable(target, requires_grad=False)
                
                data, suc, fai, max_i = self.i_fgsm_attack(x, y, epsilon, iteration, threshold=1)
                if fai == 1: print('fail at', idx)
                success += suc
                fail += fai
                if max_i > max_iter:
                    max_iter = max_i
                if max_i == 0:
                    num_of_zeros += 1
            
            # append modified data no matter success or not
            x_adv_final = data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            x_adv_final = torch.clamp(x_adv_final, 0, 1)
            x_adv_final = x_adv_final * 255
            x_adv_final = x_adv_final.squeeze().detach().cpu().numpy()
            x_adv_final = np.transpose(x_adv_final, (1, 2, 0))
            adv_result.append(x_adv_final)
            
            idx += 1

        print('max iter = {}, # of images that attacked only once = {}'.format(max_iter, num_of_zeros))
        final_acc = fail / 200
        print('wrong({}), fail({}), success({})'.format(wrong, fail, success), end='  ')
        print("Epsilon: {}  Success rate = {}\n".format(epsilon, 1 - final_acc))
        return adv_result, final_acc

"""Main"""
if __name__ == '__main__':
    # read in corresponding label
    df = pd.read_csv(label_fname)
    df = df.loc[:, 'TrueLabel'].to_numpy()
    # new an Attacker class
    attacker = Attacker(input_img_dir, df)

    epsilon = 0.01
    iteration = 50
    result, acc = attacker.attack(epsilon, iteration)

    # save images
    print('num of images =', len(result))
    for idx in range(200):
        if idx < 10:
            filename = '00{}'.format(idx)
        else: 
            if idx < 100:
                filename = '0{}'.format(idx)
            else:
                filename = idx

        if idx % 40 == 0: print('saving image', filename)
        im = Image.fromarray(np.uint8(result[idx]))
        im.save(os.path.join(output_img_dir, '{}.png'.format(filename)))