import albumentations as albu 
from PIL import Image
import torch
import pretrainedmodels
import pytorch_lightning as pl
from cnn_finetune import make_model
import torch.nn as nn
from tqdm import tqdm
import os
from torchvision.transforms import transforms
import torch.nn.functional as F
import pandas as pd
import yaml
from datetime import datetime
from src.models.se_resnext import siim_Model
from src.dataset import train_dataset, val_dataset
from src.utils import checkpoint_callback, early_stop_callback, neptune_logger, load_config

cfg = load_config('/home/chandanv/Drive/Competitions/Kaggle/SIIM/SIIM_ISIC/config.yml')
hparams = cfg['neptune_logger']['logging_params']

model = siim_Model(train_dataset = train_dataset, val_dataset= val_dataset, logger = neptune_logger, hparams = hparams)

PATH = os.path.join(cfg['neptune_logger']['model_checkpoint_params']['models_save_path'], '{}.ckpt'.format(cfg['neptune_logger']['model_checkpoint_params']['prediction_model_name']))
new_model = model.load_from_checkpoint(checkpoint_path = PATH,
                                       train_dataset = train_dataset, 
                                       val_dataset = val_dataset, 
                                       logger = neptune_logger, 
                                       hparams = hparams)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

def prediction(test_img_path, model, device):
    id_list = []
    pred_list = []
#
    with torch.no_grad():
        for image_name in tqdm(os.listdir(test_img_path)):
            #print(image_name)
            #print(os.path.join(test_img_path, image_name))
            # Preprocessing  #########################################
            img = Image.open(os.path.join(test_img_path, image_name))
            #print(img.size())
            _id = image_name.split('.')[0]
            #print('id is:',_id)

            ImageTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                               ])
            img = ImageTransform(img)
            img = img.unsqueeze(0)
            img = img.to(device)
            #print('Image shape', img.shape)
            # Predict  ##############################################
            model.eval()

            outputs = model(img)
            #print('output is:', outputs)
            #preds = F.softmax(outputs, dim=1)
            preds = nn.Sigmoid()(outputs).item() 
            #preds = preds.data
            #print('preds', preds)

            id_list.append(_id)
            pred_list.append(preds)

    # Result DataFrame
    res = pd.DataFrame({
        'image_name': id_list,
        'target': pred_list
    })
    
    # Submit
    res.sort_values(by='image_name', inplace=True)
    res.reset_index(drop=True, inplace=True)
    res.to_csv('submission.csv', index=False)
    
    return res


file_name = '{}_{}'.format(cfg['neptune_logger']['logging_params']['model_name'], datetime.now())
submission = prediction(os.path.join(cfg['neptune_logger']['logging_params']['data_dir'], 'test/'), model, device)
submission.to_csv(os.path.join(cfg['neptune_logger']['logging_params']['submission_dir'], f'{file_name}.csv'), index = False)