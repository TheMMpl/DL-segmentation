import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px  # noqa:  F401
import plotly.graph_objs as go
import seaborn as sn
import wandb
from dl_segmentation.model.model import LightningModel
from torchvision.datasets import Cityscapes
from dl_segmentation.pipelines.train import CityScapesTransform
from torchvision import transforms
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import os

# This function uses plotly.express
def check_model_inference(preprocessed_shuttles: pd.DataFrame):
    demo_path=Path('demo_results/overfit_test37')
    os.mkdir(demo_path)
    return (
        preprocessed_shuttles.groupby(["shuttle_type"])
        .mean(numeric_only=True)
        .reset_index()
    )


# This function uses plotly.graph_objects
def compare_passenger_capacity_go(preprocessed_shuttles: pd.DataFrame):
    # reference can be retrieved in artifacts panel
    # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")

    #noted checkpoints
    #checkpoint_reference = "dlprojekt/DL_segmenation/model-jql8pobq:v20"
    #checkpoint_reference='dlprojekt/DL_segmenation/model-6cwa5z66:v92'
    #checkpoint_reference='dlprojekt/DL_segmenation/model-6cwa5z66:v26'

    #larger images
    #checkpoint_reference='dlprojekt/DL_segmenation/model-daeah9nu:v23'
    checkpoint_reference='dlprojekt/DL_segmenation/model-daeah9nu:v37'

    #model_name="model-jql8pobq:v20"
    # download checkpoint locally (if not already cached)
    run = wandb.init(project="DL_segmenation")
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()
    print(artifact_dir)
    # load checkpoint
    model = LightningModel.load_from_checkpoint(Path(artifact_dir) / "model.ckpt",num_classes=34)
    model.eval()
    model.unet.eval()
    val_dataset = Cityscapes('./dataset/cityscapes', split='val',target_type='semantic', transforms = CityScapesTransform())
    train_dataset = Cityscapes('./dataset/cityscapes', split='train',
                     target_type='semantic', transforms = CityScapesTransform())
    #necessarry because we don't use the trainer here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trans = [transforms.v2.ToDtype(torch.float32, scale=True), transforms.v2.Resize((256,512))]
    jank_iter=0
    for img,target in val_dataset:
        jank_iter+=1
        for t in trans:
            img=t(img)
        image=img.to(device)
        image.unsqueeze_(0)
        result=torch.argmax(model(image)[0],dim=0).cpu().detach().numpy()
        comparison=np.vstack([result,target[0]])
        plt.imsave(f'demo_results/overfit_test37/res{jank_iter}.jpg',comparison)
        if jank_iter>50:
            break
        #plt.imsave(f'demo_results/overfit_test2/img{jank_iter}.jpg',torch.permute(img,(1,2,0)).numpy()/255)

    jank_iter=0
    for img,target in train_dataset:
        jank_iter+=1
        for t in trans:
            img=t(img)
        image=img.to(device)
        image.unsqueeze_(0)
        result=torch.argmax(model(image)[0],dim=0).cpu().detach().numpy()
        comparison=np.vstack([result,target[0]])
        plt.imsave(f'demo_results/overfit_test37/train_res{jank_iter}.jpg',comparison)  
        if jank_iter>50:
            break

    data_frame = (
        preprocessed_shuttles.groupby(["shuttle_type"])
        .mean(numeric_only=True)
        .reset_index()
    )
    fig = go.Figure(
        [
            go.Bar(
                x=data_frame["shuttle_type"],
                y=data_frame["passenger_capacity"],
            )
        ]
    )

    return fig


def create_confusion_matrix(companies: pd.DataFrame):
    actuals = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    predicted = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1]
    data = {"y_Actual": actuals, "y_Predicted": predicted}
    df = pd.DataFrame(data, columns=["y_Actual", "y_Predicted"])
    confusion_matrix = pd.crosstab(
        df["y_Actual"], df["y_Predicted"], rownames=["Actual"], colnames=["Predicted"]
    )
    sn.heatmap(confusion_matrix, annot=True)
    return plt
