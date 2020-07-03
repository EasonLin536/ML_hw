"""hw7_Network_Compression (Weight Quantization)"""
# use small model, which already performed Knowledge Distillation, for Quantization

"""Import Packages"""
import os
import sys
import torch
import pickle
import torch.nn as nn
import numpy as np
from PIL import Image
from glob import glob
import torchvision.models as models
import torchvision.transforms as transforms
from hw7_architecture_design import StudentNet # model

"""Path Specification"""
in_model_fname  = './student_model.bin'
model_8b_fname  = sys.argv[1]

"""32-bit Tensor -> 16-bit"""
def encode16(params, fname):
    """
    Args:
      	params: model's state_dict
      	fname:  filename after compression
    """
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        # some param. aren't ndarray, no compression needed
        if type(param) == np.ndarray:
            custom_dict[name] = np.float16(param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))

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

"""32-bit Tensor -> 8-bit"""
# W' = round((W - min(W)) / (max(W) - min(W)) * (2^8 - 1))
def encode8(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))

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
print('=================== Weight Quantization ===================')
print(f'original cost: {os.stat(in_model_fname).st_size} bytes.')
params = torch.load(in_model_fname)

# 32b -> 16b
# encode
# encode16(params, model_16b_filename)
# print(f'16-bit cost: {os.stat(model_16b_filename).st_size} bytes')

# 32b -> 8b
# encode
encode8(params, model_8b_fname)
print(f'8-bit cost: {os.stat(model_8b_fname).st_size} bytes')