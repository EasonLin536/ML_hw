"""hw7_Network_Compression (Knowledge Distillation)"""

"""Import Packages"""
import numpy as np
import torch
import os
import re
import sys
import pickle
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from hw7_architecture_design import StudentNet # model

"""Path Specification"""
data_dir      = sys.argv[1]
predict_fname = sys.argv[2]
model_fname   = './model_best.bin'

"""Data Processing"""
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in sorted(glob(folderName + '/*.jpg')):
            try:
                # Get classIdx by parsing image path
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            except:
                # if inference mode (there's no answer), class_idx default 0
                class_idx = 0
 
            image = Image.open(img_path)
            # Get File Descriptor
            image_fp = image.fp
            image.load()
            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            self.data.append(image)
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def get_dataloader(mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        os.path.join(data_dir, mode),
        transform=trainTransform if mode == 'training' else testTransform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader

"""WQ Decode"""
def decode16(fname):
    """
    Args:
      	fname: filename after compression
    """
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        param = torch.tensor(param)
        custom_dict[name] = param

    return custom_dict

def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param

    return custom_dict

"""Main"""
# testing
print('Testing')
print('Pre-processing testing data')
test_dataloader = get_dataloader('testing', batch_size=32)

print('Loading student model')
student_net = StudentNet(base=16).cuda()
student_net.load_state_dict(decode8(model_fname))

print('Predicting')
student_net.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        test_pred = student_net(data[0].cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

print('Writing to csv file')
with open(predict_fname, 'w') as f:
    f.write('id,label\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))