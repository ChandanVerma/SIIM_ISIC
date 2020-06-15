import warnings
warnings.filterwarnings('ignore')
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import random as rn
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import transforms
from src.augmentations import simple_augment, hard_augment

import albumentations as albu 
import pretrainedmodels

from PIL import Image
from datetime import datetime
from collections import OrderedDict
import yaml
from pytorch_lightning import seed_everything
seed_everything(2020)

def load_config(file_path):

    with open(file_path, 'r') as f:
        cfg = yaml.load(f)

    return cfg

cfg = load_config('/home/chandanv/Drive/Competitions/Kaggle/SIIM/SIIM_ISIC/config.yml')

df = pd.read_csv(os.path.join(cfg['neptune_logger']['logging_params']['data_dir'],'train_new.csv'))

X_train, X_test, y_train, y_test = train_test_split(df, df['target'], test_size = 0.1, random_state = 2020, stratify = df['target'])
X_train.reset_index(inplace = True, drop = True)
X_test.reset_index(inplace = True, drop = True)
y_train.reset_index(inplace = True, drop = True)
y_test.reset_index(inplace = True, drop = True)

class siim_dataset(Dataset):
    def __init__(self, X, y, data_dir = None, transforms = None, mode = None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.X = X
        self.Y = y
        self.mode = mode
#
    def __len__(self):
        return self.X.shape[0]
#
    def __getitem__(self, idx):
        if self.mode == 'train':
            label = self.Y.values[idx]
            image_names = self.X.image_name.values[idx]
            img_path = self.data_dir + image_names + '.jpg'
            image = Image.open(img_path)
            image = image.convert('RGB')
            image = self.transforms(image)
            return image, torch.tensor(label)
        elif self.mode == 'val':
            label = self.Y.values[idx]
            image_names = self.X.image_name.values[idx]
            img_path = self.data_dir + image_names + '.jpg'
            image = Image.open(img_path)
            image = image.convert('RGB')
            image = self.transforms(image)
            return image, torch.tensor(label)


train_class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
train_weight = 1. / train_class_sample_count
train_samples_weight = np.array([train_weight[t] for t in y_train])
train_samples_weight = torch.from_numpy(train_samples_weight)
train_sampler = WeightedRandomSampler(weights=train_samples_weight.type(torch.DoubleTensor), num_samples=len(train_samples_weight))

val_class_sample_count = np.array([len(np.where(y_test==t)[0]) for t in np.unique(y_test)])
val_weight = 1. / val_class_sample_count
val_samples_weight = np.array([val_weight[t] for t in y_test])
val_samples_weight = torch.from_numpy(val_samples_weight)
val_sampler = WeightedRandomSampler(weights=val_samples_weight.type(torch.DoubleTensor), num_samples=len(val_samples_weight))

cfg['neptune_logger']['logging_params']['augmentation']
train_dataset = siim_dataset(X_train, 
                            y_train, 
                            data_dir= cfg['neptune_logger']['logging_params']['data_dir'] + 'train/', 
                            transforms = simple_augment() if cfg['neptune_logger']['logging_params']['augmentation'] == 'simple' else hard_augment(), 
                            mode= 'train')

val_dataset = siim_dataset(X_test, 
                           y_test, 
                           data_dir= cfg['neptune_logger']['logging_params']['data_dir'] + 'train/', 
                           transforms= simple_augment(), 
                           mode= 'val')
