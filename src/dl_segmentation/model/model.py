import pytorch_lightning as L
import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image

weights=DeepLabV3_ResNet50_Weights.DEFAULT
model=deeplabv3_resnet50(weights=weights)