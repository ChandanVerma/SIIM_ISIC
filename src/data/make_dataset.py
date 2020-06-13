# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from typing import Callable, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import os
import torch
import catalyst
from catalyst.dl import utils
is_alchemy_used = True
import configparser
import yaml
from catalyst.utils import create_dataset, create_dataframe, get_dataset_labeling, map_dataframe

with open('/home/chandanv/Drive/Competitions/Kaggle/SIIM/SIIM_ISIC/src/features/config.yaml', 'r') as f:
    cfg = yaml.load(f)

train_dir = cfg['path']['data_dir'] + 'train'
dataset = create_dataset(dirs= f'{train_dir}/*', extension= '.jpg')
df = create_dataframe(dataset, col)

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.02)