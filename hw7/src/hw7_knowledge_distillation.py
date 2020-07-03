"""hw7_Network_Compression (Knowledge Distillation)"""

"""Import Packages"""
import numpy as np
import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torchvision.models as models
import re
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from hw7_architecture_design import StudentNet # model

"""Path Specification"""
data_dir            = sys.argv[1]
teacher_model_fname = './teacher_resnet18.bin'
student_model_fname = './student_model.bin'

"""Knowledge Distillation Loss"""
# Loss = alpha * T^2 * KL(Teacher's Logits / T || Student's Logits / T) + (1-alpha)(original Loss)
def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # ordinary Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # KL Divergence
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss

"""Data Processing"""
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in glob(folderName + '/*.jpg'):
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

"""Training"""
def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # initialize optimizer
        optimizer.zero_grad()
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # TeacherNet no need to backprop -> torch.no_grad
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            # use the loss the combines soft label & hard label
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()    
        else:
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num

"""Main"""
print('=================== Knowledge Distillation ===================')
# training
# pre-processing
print('Pre-processing training data')
train_dataloader = get_dataloader('training', batch_size=32)
valid_dataloader = get_dataloader('validation', batch_size=32)

print('Loading models')
teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
teacher_net.load_state_dict(torch.load(teacher_model_fname))
print(f'original cost: {os.stat(teacher_model_fname).st_size} bytes.')

student_net = StudentNet(base=16).cuda()
total = sum(p.numel() for p in student_net.parameters())
trainable = sum(p.numel() for p in student_net.parameters() if p.requires_grad)
print('\nparameter total:{}, trainable:{}\n'.format(total, trainable))

print("SGD optimizer")
optimizer = optim.SGD(student_net.parameters(), lr=1e-2, momentum=0.9)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
# print("Adam optimizer")
# optimizer = optim.Adam(student_net.parameters(), lr=1e-3)

print('Training')
teacher_net.eval() # TeacherNet is always Eval mode
now_best_acc = 0
for epoch in range(300):
    student_net.train()
    train_loss, train_acc = run_epoch(train_dataloader, update=True)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(valid_dataloader, update=False)

    # save best model
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), student_model_fname)
    print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
        epoch, train_loss, train_acc, valid_loss, valid_acc))

print(f'model size: {os.stat(student_model_fname).st_size} bytes')