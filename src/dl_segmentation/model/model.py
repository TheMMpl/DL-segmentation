import pytorch_lightning as L
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image
from torch import optim
import torch.nn.functional as F
import wandb


class ResUnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer0=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        )
        self.sconv=nn.Conv2d(3,64,kernel_size=1,stride=1,padding=0)
        self.layer1=ResConv(64,128,stride=2)
        self.layer2=ResConv(128,256,stride=2)
        self.bridge=ResConv(256,512,stride=2,residual=False)
        self.up1=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.layer4=ResConv(512,256,stride=1,skip=True)
        self.up2=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.layer5=ResConv(256,128,stride=1,skip=True)
        self.up3=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.layer6=ResConv(128,64,stride=1,skip=True)
        self.unmask=nn.Sequential(
            nn.Conv2d(64,num_classes,kernel_size=1,stride=1),
        )
        #czemu 1x1 conv- żeby dopasować ilość kanałów - bo robimy res connection ale z inchannenl trzeba zrobić outchannels
    
    def forward(self,x):
        skip1=self.layer0(x)+self.sconv(x)
        skip2=self.layer1(skip1)
        skip3=self.layer2(skip2)
        x=self.bridge(skip3)
        x=self.up1(x)
        x = torch.cat([x, skip3], dim=1)
        x=self.layer4(x)
        x=self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x=self.layer5(x)
        x=self.up3(x)
        x = torch.cat([x, skip1], dim=1)
        x=self.layer6(x)
        return self.unmask(x)

class ResConv(nn.Module):
    def __init__(self, channels_in, channels_out, stride,residual=True,skip=False):
        super().__init__()
        self.residual=residual
        
        #we downsample once per block
        if skip:
            skip_channels=channels_out
        else:
            skip_channels=0
        self.conv_unit=nn.Sequential(
            nn.BatchNorm2d(channels_in+skip_channels),
            nn.GELU(),
            nn.Conv2d(channels_in+skip_channels,channels_out,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(channels_out),
            nn.GELU(),
            nn.Conv2d(channels_out,channels_out,kernel_size=3,stride=1,padding=1),
        )
        self.skip_conv=nn.Conv2d(channels_in+skip_channels,channels_out,kernel_size=1,padding=0,stride=stride)

    def forward(self,x):
        if self.residual:
            return self.skip_conv(x)+self.conv_unit(x)
        else:
            return self.conv_unit(x)

class LightningModel(L.LightningModule):
    def __init__(self, num_classes,):
        super().__init__()
        self.unet=ResUnet(num_classes)
        self.lossfunc=nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x,y=batch
        x=self.unet(x)
        y.squeeze_()
        loss=self.lossfunc(x,y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y=batch
        pred=self.unet(x)
        y.squeeze_()
        loss=self.lossfunc(pred,y)
        self.log("val_loss", loss)
        #return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.encoder(x)
        return pred
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        return self.unet.forward(x)
    
    def test_step(self, batch, batch_idx):
        x,y = batch
        pred = self.unet(x)
        y.squeeze_()
        loss=self.lossfunc(pred,y)
        self.log("test_loss", loss)
        return loss
