import pytorch_lightning as L
import torch
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from torchvision.io.image import read_image
#from dl_segmentation.model import ResUnet
from dl_segmentation.model.model import LightningModel

unet=LightningModel(20)
print(unet.parameters())
#trainer = L.Trainer(limit_train_batches=2, max_epochs=1)
#train_loader=DataLoader(Cityscapes)