import pytorch_lightning as L
from dl_segmentation.model.model import LightningModel
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
import torch
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

IMG_SIZE=256

class CityScapesTransform:
    def __call__(self, image, target):
        trans_im = [transforms.PILToTensor(), transforms.v2.ToDtype(torch.float32, scale=True), transforms.v2.Resize((IMG_SIZE,IMG_SIZE*2))]
        trans_tar = [transforms.PILToTensor(), transforms.v2.ToDtype(torch.long, scale=False), transforms.v2.Resize((IMG_SIZE,IMG_SIZE*2),interpolation=transforms.InterpolationMode.NEAREST)]
        for t in trans_im:
            image = t(image)
        for t in trans_tar:
            target = t(target)
        return image, target

class ExtraLabelsTransform:
    def __call__(self, image, target):
        weights=DeepLabV3_ResNet101_Weights.DEFAULT
        trans_im=weights.transforms()
        trans_tar = [transforms.PILToTensor(), transforms.v2.ToDtype(torch.long, scale=False), transforms.v2.Resize((IMG_SIZE,IMG_SIZE*2),interpolation=transforms.InterpolationMode.NEAREST)]
        image=trans_im(image)
        for t in trans_tar:
            target = t(target)
        return image, target

# To wszystko co tu było jest w data_science.nodes.py


