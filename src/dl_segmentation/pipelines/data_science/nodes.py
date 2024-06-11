import logging
from typing import Dict, Tuple
import os

import torch
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pytorch_lightning as L
from dl_segmentation.model.model import LightningModel
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from dl_segmentation.pipelines.train import CityScapesTransform
from torchvision import transforms
from torchvision.transforms import v2
import torch
from matplotlib import pyplot as plt
from consts import BATCH_SIZE, MAX_EPOCHS, NUM_WORKERS, LOG_STEPS, NUM_CLASSES


def train_model():

    unet=LightningModel(NUM_CLASSES)
    
    wandb_logger = WandbLogger(project="DL_segmenation",log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",save_top_k=10,every_n_epochs=1)
    test_callback = ModelCheckpoint(monitor="train_loss",save_top_k=5,every_n_epochs=1)
    trainer = Trainer(logger=wandb_logger,callbacks=[checkpoint_callback,test_callback],max_epochs=MAX_EPOCHS,log_every_n_steps=LOG_STEPS)
    train_dataset = Cityscapes('./dataset/cityscapes', split='train',
                     target_type='semantic', transforms = CityScapesTransform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    trainer = Trainer(logger=wandb_logger,callbacks=[checkpoint_callback],max_epochs=MAX_EPOCHS,log_every_n_steps=LOG_STEPS)
    val_dataset = Cityscapes('./dataset/cityscapes', split='val',
                     target_type='semantic', transforms = CityScapesTransform())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    trainer.fit(unet,train_loader,val_loader)
    
    return unet, trainer


def test_model(unet, trainer):
    test_dataset = Cityscapes('./dataset/cityscapes', split='test',
                     target_type='semantic', transforms = CityScapesTransform())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    trainer.test(model=unet,dataloaders=test_loader)
    
    return {"A":0.}

