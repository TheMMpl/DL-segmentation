import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px  # noqa:  F401
import plotly.graph_objs as go
import seaborn as sn
import wandb
from dl_segmentation.model.model import LightningModel
from torchvision.datasets import Cityscapes
from dl_segmentation.pipelines.train import CityScapesTransform
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision import transforms
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import os
from consts import NUM_CLASSES, IMG_SIZE, MODEL_CHECKPOINT
from PIL import Image

# used when loading a single image
def prepare_data(imagepaths):
    transform = transforms.v2.Compose([transforms.PILToTensor(), transforms.v2.ToDtype(torch.float32, scale=True), transforms.v2.Resize((IMG_SIZE,IMG_SIZE*2))])
    images=[Image.open(img) for img in imagepaths]
    return [transform(img) for img in images]

'''
evaluation pipeline
'''

def load_model(checkpoint):
    run = wandb.init(project="DL_segmenation")
    artifact = run.use_artifact(checkpoint, type="model")
    artifact_dir = artifact.download()
    model = LightningModel.load_from_checkpoint(Path(artifact_dir) / "model.ckpt",num_classes=NUM_CLASSES)
    model.eval()
    model.unet.eval()
    return model

def prepare_dataset(image_amount):
    val_dataset = Cityscapes('./dataset/cityscapes', split='val',target_type='semantic', transforms = CityScapesTransform())
    train_dataset = Cityscapes('./dataset/cityscapes', split='train',
                     target_type='semantic', transforms = CityScapesTransform())
    transform=transforms.v2.Compose([transforms.v2.ToDtype(torch.float32, scale=True), transforms.v2.Resize((256,512))])
    val_snippet=[(transform(img),target) for i,(img,target) in enumerate(val_dataset) if i<image_amount]
    train_snippet=[(transform(img),target) for i,(img,target) in enumerate(train_dataset) if i<image_amount]
    return val_snippet #, train_snippet

def run_inference(model,data):
    result=[]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for img in data:
        if type(img) is tuple:
            image=img[0].to(device)
            image.unsqueeze_(0)
            output=torch.argmax(model(image)[0],dim=0).cpu().detach().numpy()
            result.append(np.vstack([output,img[1][0]]))
        else:
            image=img.to(device)
            image.unsqueeze_(0)
            result.append(torch.argmax(model(image)[0],dim=0).cpu().detach().numpy())
    return result

def save_results(results):
    Path(f"demo_results/{MODEL_CHECKPOINT}").mkdir(parents=True, exist_ok=True)
    metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES)
    for i,res in enumerate(results):
        plt.imsave(f'demo_results/{MODEL_CHECKPOINT}/res{i}.jpg',res)
    return results



