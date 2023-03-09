import os 
import argparse
import pytorch_lightning as pl

# when using 
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from datasets import *

# for pytorch lightning test
# todo : create classification pytorch lightning module which to customize or evaluate backbones.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

class ResNetClassifier(pl.LightningModule):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    
    optimizers = {"adam": Adam, "sgd": SGD}
    
    def __init__(self, num_classes, resnet_version, train_path, val_path, test_path=None, optimizer="adam", lr=1e-3, batch_size=16, transfer=True, tune_fc_only=False):
        super(ResNetClassifier, self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = self.optimizers[optimizer]
        self.loss_fn = (
            nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        )
        self.acc = Accuracy(
            task="binary" if num_classes == 1 else "multiclass", num_classes = num_classes
        )
        self.resnet_model = self.resnets[resnet_version](pretrained=transfer)
        linear_size = list(self.resnet_model.children())[-1].in_features
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)
        
        # based on https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py

