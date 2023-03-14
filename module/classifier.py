import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torchmetrics import Accuracy

class Classifier(pl.LightningModule):
    optimizers = {
        'adam' : Adam,
        'sgd' : SGD
    }
    def __init__(self, model, cfg):
        super(Classifier, self).__init__()
        self.model = model
        
        self.num_classes = cfg['classes']
        self.loss_fn = (
            nn.BCEWithLogitsLoss() if self.num_classes == 1 else nn.CrossEntropyLoss()
        )
        self.acc = Accuracy(
            multiclass=False if self.num_classes == 1 else True, num_classes=self.num_classes
        )
        self.optimizer = self.optimizers[cfg['optimizer']]
        self.optimizer_opt = cfg['optimizer_options']
        # if you ignore several options, input type of list
        self.save_hyperparameters(ignore='model')
    
    def _step(self, batch):
        x, y = batch
        preds = self.model(x)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()
        y = y.type(torch.LongTensor)
        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.optimizer_opt)
    