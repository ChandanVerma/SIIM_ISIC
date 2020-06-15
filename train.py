from pytorch_lightning import Trainer
from src.models.se_resnext import siim_Model
from src.dataset import train_dataset, val_dataset
from src.utils import checkpoint_callback, early_stop_callback, neptune_logger
from src.utils import load_config, dict_to_args
import yaml
from pytorch_lightning import seed_everything
seed_everything(2020)

cfg = load_config('/home/chandanv/Drive/Competitions/Kaggle/SIIM/SIIM_ISIC/config.yml')

def main(hparams):

    ## initilize lightning model 
    model = siim_Model(train_dataset, val_dataset, neptune_logger, hparams)

    ## init trainer
    trainer = Trainer(
        max_epochs = hparams.epochs,
        gpus = hparams.gpus,
        batch_size = hparams.batch_size, 
        distributed_backend = hparams.distributed_backend,
        precision = 16 if hparams.use_16bit else 32,
        checkpoint_callback= checkpoint_callback,
        logger = neptune_logger, 
        early_stop_callback = early_stop_callback,
        profiler = True
        )

    ## start training
    trainer.fit(model)     

if __name__ == '__main__':

    hparams = dict_to_args(cfg['neptune_logger']['logging_params'])
    main(hparams)
    neptune_logger.experiment.log_artifact(cfg['neptune_logger']['logging_params']['artifacts_dir'])
    neptune_logger.experiment.stop()
