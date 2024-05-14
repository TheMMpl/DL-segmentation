import pytorch_lightning as L
import torch
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from torchvision.io.image import read_image
from torchmetrics.classification import MulticlassJaccardIndex
#from dl_segmentation.model import ResUnet
from dl_segmentation.model.model import LightningModel
import pytest


data_params=[]
inference_params=[]
@pytest.mark.parametrize("results_path,dataset_path",data_params)
def prepare_data(results_path,dataset_path):
    pass


@pytest.mark.parametrize("model_path,results_path,dataset_path,test_size,metric",inference_params)
def prepare_model():
    pass
 

@pytest.mark.parametrize("model_path,results_path,dataset_path,test_size,metric",inference_params)
def test_model(model_path,results_path,dataset_path,test_size,metric):
    

def test_Unet(model_path,results_path,dataset_path,test_size,metric):

#trainer = L.Trainer(limit_train_batches=2, max_epochs=1)
#train_loader=DataLoader(Cityscapes)