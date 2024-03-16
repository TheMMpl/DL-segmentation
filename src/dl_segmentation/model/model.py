import pytorch_lightning as L
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image
import torch.nn.functional as F

weights=DeepLabV3_ResNet50_Weights.DEFAULT
model=deeplabv3_resnet50(weights=weights)

class ResUnet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer0=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(64),
            nn.Conv2d(64,64)
        )
        self.sconv=nn.Conv2d(3,64,kernel_size=1,stride=2,padding=0)
        self.layer1=ResConv()
        self.layer2=ResConv()
        self.bridge=ResConv(residual=False)
        self.up1=nn.Upsample()
        self.layer4=ResConv()
        self.up2=nn.Upsample()
        self.layer5=ResConv()
        self.up3=nn.Upsample()
        self.layer6=ResConv()
        self.unmask=nn.Sequential(
            nn.Conv2d(),
            nn.Softmax2d()
        )

        #czemu 1x1 conv- żeby dopasować ilość kanałów - bo robimy res connection ale z inchannenl trzeba zrobić outchannels
        #jakie loss function?- chyba cross entropy - czy log nie wiem


    
    def forward(self,x):
        skip1=self.layer0(x)+self.sconv(x)
        skip2=self.layer1(skip1)
        skip3=self.layer2(skip2)
        x=self.bridge(skip3)
        x=self.up1(x)
        x=self.layer4(x,skip1)
        x=self.layer5(x,skip2)
        x=self.layer6(x,skip3)
        return self.unmask(x)

class ResConv(nn.Module):
    def __init__(self, channels_in, channels_out, stride,residual=True) -> None:
        super().__init__()
        self.residual=residual
        #we downsample once per block
        self.conv_unit=nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.GELU(),
            nn.Conv2d(channels_in,channels_out,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(channels_out),
            nn.GELU(),
            nn.Conv2d(channels_out,channels_out,kernel_size=3,stride=stride,padding=1),
        )
        self.skip_conv=nn.Conv2d(channels_in,channels_out,kernel_size=1,padding=0,stride=stride)
    def forward(self,x):
        if self.residual:
            return self.skip_conv(x)+self.conv_unit(x)
        else:
            return self.conv_unit(x)
