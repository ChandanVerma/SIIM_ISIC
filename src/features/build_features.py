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

with open('/home/chandanv/Drive/Competitions/Kaggle/SIIM/SIIM_ISIC/src/features/config.yaml', 'r') as f:
    cfg = yaml.load(f)

print(f'torch: {torch.__version__} , catalyst: {catalyst.__version__}')
SEED = 2020

utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic= True)

if is_alchemy_used:
    monitoring_params = {
        "token": "0b5b046a75bdc36fdb8096e60df70082",
        "project": "SIIM_ISIC",
        "group": "first_trials",
        "experiment": "first_experiment" 
    }
    assert monitoring_params['token'] is not None
else:
    monitoring_params = None


