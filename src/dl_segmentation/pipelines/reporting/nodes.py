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
    #train_dataset = Cityscapes('./dataset/cityscapes', split='train',target_type='semantic', transforms = CityScapesTransform())
    transform=transforms.v2.Compose([transforms.v2.ToDtype(torch.float32, scale=True), transforms.v2.Resize((256,512))])
    val_snippet=[(transform(img),target) for i,(img,target) in enumerate(val_dataset) if i<image_amount]
    image_snippet=[img for i,(img,target) in enumerate(val_dataset) if i<image_amount]
    #train_snippet=[(transform(img),target) for i,(img,target) in enumerate(train_dataset) if i<image_amount]
    return val_snippet, image_snippet#, train_snippet

def run_inference(model,data):
    results=[]
    IoU=[]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES).to(device)
    for img in data:
        if type(img) is tuple:
            image=img[0].to(device)
            image.unsqueeze_(0)
            output=model(image)[0]
            pred=output.view(output.size(0),-1).unsqueeze_(0)
            target_tensor=img[1][0].flatten().unsqueeze_(0).to(device)
            IoU.append(metric(pred,target_tensor).cpu().detach().numpy())
            result=torch.argmax(output,dim=0).cpu().detach().numpy()
            results.append(np.vstack([result,img[1][0]]))
        else:
            image=img.to(device)
            image.unsqueeze_(0)
            results.append(torch.argmax(model(image)[0],dim=0).cpu().detach().numpy())
    return results, IoU

def save_results(results,images,IoU):
    metrics=pd.DataFrame(IoU,columns=['IoU'])
    Path(f"demo_results/{MODEL_CHECKPOINT}").mkdir(parents=True, exist_ok=True)
    for i,res in enumerate(results):
        plt.imsave(f'demo_results/{MODEL_CHECKPOINT}/res{i}.jpg',res)
    for i, img in enumerate(images):
        plt.imsave(f'demo_results/{MODEL_CHECKPOINT}/img{i}.jpg',torch.permute(img,(1,2,0)).cpu().detach().numpy())
    return results, metrics



