from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from torchvision.transforms import transforms
import sys
import yaml

def load_config(file_path):

    with open(file_path, 'r') as f:
        cfg = yaml.load(f)

    return cfg

cfg = load_config('/home/chandanv/Drive/Competitions/Kaggle/SIIM/SIIM_ISIC/config.yml')

checkpoint_callback = ModelCheckpoint(
    filepath= cfg['neptune_logger']['model_checkpoint_params']['models_save_path'],
    save_weights_only= cfg['neptune_logger']['model_checkpoint_params']['save_weights_only'],
    save_top_k= cfg['neptune_logger']['model_checkpoint_params']['save_top_k'],
    verbose= cfg['neptune_logger']['model_checkpoint_params']['verbose'],
    monitor= cfg['neptune_logger']['model_checkpoint_params']['monitor'],
    mode = cfg['neptune_logger']['model_checkpoint_params']['mode']
)

early_stop_callback = EarlyStopping(
    monitor = cfg['neptune_logger']['early_stop_params']['monitor'],
    min_delta = cfg['neptune_logger']['early_stop_params']['min_delta'],
    patience = cfg['neptune_logger']['early_stop_params']['patience'],
    verbose = cfg['neptune_logger']['early_stop_params']['verbose'],
    mode = cfg['neptune_logger']['early_stop_params']['mode']
)

neptune_logger = NeptuneLogger(
    api_key = cfg['neptune_logger']['api_params']['api_key'],
    project_name = cfg['neptune_logger']['api_params']['project_name'],
    params = cfg['neptune_logger']['logging_params'],
    experiment_name = cfg['neptune_logger']['api_params']['experiment_name'],
    close_after_fit = cfg['neptune_logger']['api_params']['close_after_fit']
) 
