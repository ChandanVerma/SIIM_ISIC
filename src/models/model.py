import torch.nn as nn
import pretrainedmodels
from cnn_finetune import make_model

def make_classifier(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 1),
    )

def get_model(model_name: str, num_classes: int, pretrained: bool = False):
    model = make_model(model_name, num_classes, 
                       pretrained = pretrained, 
                       input_size=(512, 512), 
                       pool=nn.AdaptiveMaxPool2d(1),
                       dropout_p= 0.2,
                       classifier_factory=make_classifier)

    return model


model_name = "resnet18"
siim_Model = get_model(model_name, 1)