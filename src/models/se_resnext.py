import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np 
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from geffnet import create_model

import pytorch_lightning as pl 
from pytorch_lightning import seed_everything
from pytorch_lightning import _logger as log 
from pytorch_lightning.core import LightningModule

import torch_optimizer as optim
from src.generalized_mean_pooling import GeM
from src.dataset import train_sampler, val_sampler
## fixing random seed
seed_everything(2020)

class siim_Model(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, logger, hparams):
        super().__init__()

        self.hparams = hparams     
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.logger = logger 

        self.model = create_model(self.hparams.model_name, num_classes = self.hparams.num_classes)
        #self.model.global_pool = GeM()
                       

    def forward(self, x):
        bs, _, _, _ = x.shape

        x = self.model(x)

        # x = self.model._features(x)
        # x = F._adaptive_max_pool2d(x, 1).reshape(bs, -1)
        # x = F.dropout(x, p = 0.5)
        # x = self.model._classifier(x)
        return x
# 
    # training setup
    def train_dataloader(self):
        log.info('Training data loader called.')
        return DataLoader(self.train_dataset, batch_size = self.hparams.batch_size, sampler = train_sampler, shuffle = False, drop_last= True, num_workers= 10)
# 
    def val_dataloader(self):
        log.info('Validation data loader called.')
        return DataLoader(self.val_dataset, batch_size = self.hparams.batch_size, sampler = val_sampler, shuffle= False, drop_last= True, num_workers= 10)
#        
    def configure_optimizers(self):
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr= self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.2)
                                                               
        return [optimizer], [scheduler]        
#
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        loss = F.binary_cross_entropy_with_logits(output, target)
        self.logger.experiment.log_metric('loss', loss.item())
        return {'loss': loss}
# 
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        val_loss = nn.BCEWithLogitsLoss()(output, target)
        probs_ = F.sigmoid(output)
        #val_loss = val_loss.unsqueeze(dim=-1)
        self.logger.experiment.log_metric('val_loss', val_loss.item())      
        return {'val_loss': val_loss, "probs_": probs_, "gt": target}
# 
    def validation_epoch_end(self, output):
        # """
        # Called at the end of validation to aggregate outputs.
        # :param outputs: list of individual outputs of each validation step.
        # """
        avg_loss = torch.stack([x['val_loss'] for x in output]).mean()
        probs = torch.stack([x['probs_'] for x in output])
        gt = torch.stack([x['gt'] for x in output])
        probs = probs.detach().cpu().numpy().reshape(1,-1).squeeze(0)
        gt = gt.detach().cpu().numpy().reshape(1,-1).squeeze(0)

        auc_roc = roc_auc_score(gt, probs)

        accuracy = accuracy_score(gt, np.around(probs))

        mlflow_logs = {'val_loss': avg_loss.item(), 'auc': round(auc_roc, 4), 'accuracy': accuracy}
        self.logger.experiment.log_metric('avg_loss', avg_loss.item())
        self.logger.experiment.log_metric('auc_score', round(auc_roc, 4))
        self.logger.experiment.log_metric('accuracy', accuracy)
        return {'val_loss': avg_loss, 'log': mlflow_logs}