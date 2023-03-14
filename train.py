import os 

import albumentations
from albumentations.pytorch.transforms import ToTensorV2

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from datasets import cifar10, tinyimagenet
from models import resnet18
from module.classifier import Classifier
from utils.yaml_helper import get_train_configs

import platform
import argparse

def train(cfg):
    train_transforms = albumentations.Compose([
        albumentations.Normalize(0, 1),
        ToTensorV2(),
    ],)
    if cfg['dataset_name'] == 'tinyimagenet':
        data_module = tinyimagenet.TinyImageNet(
            path=cfg['data_path'],
            workers=cfg['workers'],
            transforms=train_transforms,
            batch_size=cfg['batch_size'],
        )
    elif cfg['dataset_name'] == 'cifar10':
        data_module = cifar10.Cifar10(
            data_path=cfg['data_path'],
            workers=cfg['workers'],
            transforms=train_transforms,
            batch_size=cfg['batch_size'],
            input_size=(56, 56)
        )
    
    model = resnet18.ResNet_FC(resnet_version='resnet18', output_channel=cfg['classes'])
    
    model_module = Classifier(model=model, cfg=cfg)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg['save_dir'],
            filename="resnet-model-{epoch}-{val_loss:.2f}-{val_acc:0.2f}",
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            save_last=True,
            every_n_epochs=cfg['save_freq']
        )
    ]
    
    trainer_args = {
        "accelerator": "ddp" if platform.system() != 'Windows' else 'auto',
        "devices": cfg['gpus'],
        "max_epochs": cfg['epochs'],
        "plugins" : DDPPlugin(find_unused_parameters=True) if platform.system() != 'Windows' else None,
        "callbacks": callbacks
    }
    
    trainer = pl.Trainer(
        **trainer_args,
        **cfg['trainer_options'])
    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./config/resnet18_cifar10.yaml', type=str)
    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)
    
    train(cfg)