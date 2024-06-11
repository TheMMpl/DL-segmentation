from typing import Dict, Tuple
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
import pandas as pd
import torch
import os
from dl_segmentation.pipelines.train import CityScapesTransform
from dl_segmentation.pipelines.train import ExtraLabelsTransform
from torchvision import transforms
import logging

from consts import DOWNLOAD_DATASET, CITYSCAPES_USERNAME, CITYSCAPES_PASSWORD

def download_data():
	try:
		if DOWNLOAD_DATASET == 1:
			logging.getLogger(__name__).info("Downloading data...")
			os.system("wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username="+CITYSCAPES_USERNAME+"&password="+CITYSCAPES_PASSWORD+"&submit=Login' https://www.cityscapes-dataset.com/login/; history -d $((HISTCMD-1))")
			os.system("wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1")
			os.system("wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3")
			logging.getLogger(__name__).info("Creating directories...")
			os.system("mkdir ./dataset")
			os.system("mkdir ./dataset/cityscapes")
			logging.getLogger(__name__).info("Unpacking...")
			os.system("unzip gtFine_trainvaltest.zip -d ./dataset/cityscapes")
			os.system("rm ./dataset/cityscapes/license.txt")
			os.system("rm ./dataset/cityscapes/README")
			os.system("unzip leftImg8bit_trainvaltest.zip -d ./dataset/cityscapes")
			os.system("rm ./dataset/cityscapes/license.txt")
			os.system("rm ./dataset/cityscapes/README")
			logging.getLogger(__name__).info("Cleaning...")
			os.system("rm ./leftImg8bit_trainvaltest.zip")
			os.system("rm ./gtFine_trainvaltest.zip")
			os.system("rm ./cookies.txt")
			logging.getLogger(__name__).info("Done!")
	except:
		return False
	return True
