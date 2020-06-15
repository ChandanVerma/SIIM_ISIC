import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np 
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from cnn_finetune import make_model

import pytorch_lightning as pl 
from pytorch_lightning import seed_everything
from pytorch_lightning import _logger as log 
from pytorch_lightning.core import LightningModule

import torch_optimizer as optim

## fixing random seed
seed_everything(2020)

class siim_Model(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, logger, hparams):
        super().__init__()

        self.hparams = hparams     
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.logger = logger 


        def make_classifier(in_features, num_classes):
            return nn.Sequential(
                nn.Linear(in_features, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, self.hparams['num_classes']),
            )

        self.model = make_model(self.hparams['model_name'], self.hparams['num_classes'], 
                       pretrained = self.hparams['pretrained'], 
                       input_size=(self.hparams['image_size'], self.hparams['image_size']), 
                       pool=nn.AdaptiveMaxPool2d(1),
                       dropout_p= 0.5,
                       classifier_factory= make_classifier)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model._features(x)
        x = F._adaptive_max_pool2d(x, 1).reshape(bs, -1)
        x = F.dropout(x, p = 0.5)
        x = self.model._classifier(x)

        # x = self.model.relu(x)
        # x = self.model.relu(x)
        # x = self.model.relu(x)
        # x = self.model.maxpool(x)
        
        # x = self.model.layer1(x)
        # x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)
        # x = F._adaptive_max_pool2d(x, 1).reshape(bs, -1)
        # x = self.logits(x)
        return x
# 
    # training setup
    def train_dataloader(self):
        log.info('Training data loader called.')
        return DataLoader(self.train_dataset, batch_size = self.hparams['batch_size'], shuffle = True, drop_last= True, num_workers= 10)
# 
    def val_dataloader(self):
        log.info('Validation data loader called.')
        return DataLoader(self.val_dataset, batch_size = self.hparams['batch_size'], shuffle= False, drop_last= True, num_workers= 10)
#        
    def configure_optimizers(self):
        optimizer = optim.RAdam(self.model.parameters(), lr= self.hparams['lr'], 
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=4e-3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams['batch_size'] , eta_min=self.hparams['lr'])
        return [optimizer], [scheduler]        
#
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        loss = F.binary_cross_entropy_with_logits(output, target)
        self.logger.experiment.log_metric('train_loss', loss.item())
        return {'loss': loss}
# 
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        val_loss = F.binary_cross_entropy_with_logits(output, target)
        probs = F.sigmoid(output)
        #val_loss = val_loss.unsqueeze(dim=-1)
        self.logger.experiment.log_metric('val_loss', val_loss.item())      
        return {'val_loss': val_loss, "probs": probs, "gt": target}
# 
    def validation_epoch_end(self, output):
        # """
        # Called at the end of validation to aggregate outputs.
        # :param outputs: list of individual outputs of each validation step.
        # """
        avg_loss = torch.stack([x['val_loss'] for x in output]).mean()
        probs = torch.stack([x['probs'] for x in output])
        gt = torch.stack([x['gt'] for x in output])
        probs = probs.detach().cpu().numpy().squeeze(2)
        gt = gt.detach().cpu().numpy().squeeze(2)

        try:
            auc_roc = roc_auc_score(gt, probs)
        except ValueError:
            auc_roc = 0

        mlflow_logs = {'val_loss': avg_loss.item(), 'auc': auc_roc}
        self.logger.experiment.log_metric('avg_loss', avg_loss.item())
        self.logger.experiment.log_metric('auc_score', auc_roc)
        return {'val_loss': avg_loss, 'log': mlflow_logs}