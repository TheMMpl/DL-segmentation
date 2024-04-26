import logging
from typing import Dict, Tuple
import os

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


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model():
    BATCH_SIZE = 32
    MAX_EPOCHS = 20
    NUM_WORKERS = 7
    LOG_STEPS = 5
    unet=LightningModel(20)

    wandb_logger = WandbLogger(project="DL_segmenation",log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",save_top_k=10,every_n_epochs=1)
    trainer = Trainer(logger=wandb_logger,callbacks=[checkpoint_callback],max_epochs=MAX_EPOCHS,log_every_n_steps=LOG_STEPS)
    train_dataset = Cityscapes('./dataset/cityscapes', split='train', mode='fine',
                     target_type='semantic', transforms = CityScapesTransform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    trainer.fit(model=unet,train_dataloaders=train_loader)


    return unet


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    me = max_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    return {"r2_score": score, "mae": mae, "max_error": me}
