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
    MAX_EPOCHS = 1
   # NUM_WORKERS = 7
    LOG_STEPS = 5
    #torch.set_float32_matmul_precision('medium' | 'high')

    unet=LightningModel(30)

    wandb_logger = WandbLogger(project="DL_segmenation",log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",save_top_k=10,every_n_epochs=1)
    trainer = Trainer(accelerator="auto",logger=wandb_logger,callbacks=[checkpoint_callback],max_epochs=MAX_EPOCHS,log_every_n_steps=LOG_STEPS)
    train_dataset = Cityscapes('./dataset/cityscapes', split='train', mode='fine',
                     target_type='semantic', transforms = CityScapesTransform())
    #i don't yet understand what multithreading does with a single cuda gpu
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    trainer.fit(model=unet,train_dataloaders=train_loader)

    # sanity check test
  
    def tensor_to_image(tensor):
        np_res=torch.argmax(tensor,dim=1).detach().numpy()
        result=np.squeeze(np_res)
        print(result.shape)
        unique_classes = np.unique(tensor)
        colors = np.random.randint(0, 256, size=(30, 3))  # Generate random colors for each class

        # Create an empty image
        image = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)

        # Assign random colors to each class
        for i, cls in enumerate(unique_classes):
            #interesting trick
            mask = result == cls
            image[mask] = colors[i]

        print(image)
        im=Image.fromarray(image)
        return im


    transform = CityScapesTransform()
    unet.eval()
    #maybe this helps?
    unet.unet.eval()
    testpaths=["frankfurt_000000_001016_leftImg8bit.jpg","frankfurt_000000_000294_leftImg8bit.jpg","frankfurt_000000_001236_leftImg8bit.jpg"]
    targetpaths=["frankfurt_000000_001016_gtFine_color.png","frankfurt_000000_000294_gtFine_color.png","frankfurt_000000_001236_gtFine_color.png"]
    testimages=[Image.open(img) for img in testpaths]
    targetimages=[Image.open(img) for img in targetpaths]
    print(targetimages[1])
    with torch.no_grad():
        for i in range(3):
            transformed_img,transformed_res=transform(testimages[i],targetimages[i])
            results=unet(transformed_img.unsqueeze(0))
            tensor_to_image(results).save("test_image.jpg")
            #print(results.shape)


    return unet, trainer


def evaluate_model(
    unet: LightningModel, trainer: Trainer
) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    BATCH_SIZE = 32
    MAX_EPOCHS = 20
    NUM_WORKERS = 7
    LOG_STEPS = 5

    test_dataset = Cityscapes('./dataset/cityscapes', split='train', mode='fine',
                     target_type='semantic', transforms = CityScapesTransform())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    trainer.test(unet,dataloaders=test_loader)
    # y_pred = regressor.predict(X_test)
    # score = r2_score(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)
    # me = max_error(y_test, y_pred)
    # logger = logging.getLogger(__name__)
    # logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    return {"ziu"}
