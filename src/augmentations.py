from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, Resize,
                            RandomBrightness, RandomContrast, HueSaturationValue, Blur, GaussNoise,
                            Rotate, RandomResizedCrop, Cutout)
from albumentations.pytorch import ToTensorV2, ToTensor
from torchvision.transforms import transforms

def hard_augment():
    hard_augment_ = transforms.Compose([
                      transforms.RandomHorizontalFlip(p=0.5),
                      transforms.RandomRotation(degrees=(-90, 90)),
                      transforms.RandomVerticalFlip(p=0.5),
                      transforms.ToTensor(),  
                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                          
                      ])
    return hard_augment_

def medium_augment():
    medium_augment_ = transforms.Compose([
                      transforms.CenterCrop((100, 100)),
                      transforms.RandomCrop((80, 80)),
                      transforms.RandomHorizontalFlip(p=0.5),
                      transforms.RandomRotation(degrees=(-90, 90)),
                      transforms.RandomVerticalFlip(p=0.5),
                      transforms.ToTensor(),
                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                      ])
    return medium_augment_

def simple_augment():
    simple_augment_ =  transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                       
                      ])
    return simple_augment_
