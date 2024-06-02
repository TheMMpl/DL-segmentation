import pytest
import wandb
import torch
from dl_segmentation.model.model import LightningModel
from pathlib import Path
from torchvision.datasets import Cityscapes
from dl_segmentation.pipelines.train import CityScapesTransform
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision import transforms
import numpy as np
from kedro.io import DataCatalog
import matplotlib.pyplot as plt
from consts import NUM_CLASSES, MODEL_CHECKPOINT

data_params=("demo_results/test_IoU",'./dataset/cityscapes')
model_checkpoint=MODEL_CHECKPOINT
test_size=10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES).to(device)


@pytest.fixture(scope="session")
def prepare_data(request):
    Path(request.param[0]).mkdir(parents=True, exist_ok=True)
    val_dataset = Cityscapes(request.param[1], split='val',target_type='semantic', transforms = CityScapesTransform())
    return val_dataset, request.param[0]


@pytest.fixture(scope="session")
def prepare_model(request):
    run = wandb.init(project="DL_segmenation")
    artifact = run.use_artifact(request.param, type="model")
    artifact_dir = artifact.download()
    print(artifact_dir)
    model = LightningModel.load_from_checkpoint(Path(artifact_dir) / "model.ckpt",num_classes=34)
    model.eval()
    model.unet.eval()
    return model
 

@pytest.mark.parametrize("test_size,metric,device,prepare_data,prepare_model",[(test_size,metric,device,data_params,model_checkpoint)],
                         indirect=["prepare_data","prepare_model"])
def test_model(test_size,metric,device,prepare_data,prepare_model):
    """
    test model inference on a number of images from the validation dataset and check the IoU metric relative to the ground truth
    """
    
    IoU=0
    model=prepare_model
    val_dataset,results_path=prepare_data
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES).to(device)
    trans = [transforms.v2.ToDtype(torch.float32, scale=True), transforms.v2.Resize((256,512))]
    image_num=0
    for img,target in val_dataset:
        image_num+=1
        for t in trans:
            img=t(img)
        image=img.to(device)
        image.unsqueeze_(0)
        output=model(image)[0]
        preds=output.view(output.size(0),-1).unsqueeze_(0)
        target_tensor=target[0].flatten().unsqueeze_(0).to(device)
        IoU+=metric(preds,target_tensor)
        result=torch.argmax(model(image)[0],dim=0).cpu().detach().numpy()
        comparison=np.vstack([result,target[0]])
        #plt.imsave(f'{results_path}/res{image_num}.jpg',comparison)
        if image_num>test_size:
            IoU=IoU/test_size  
            assert IoU>=0.3
            # f = open(f'{results_path}/IoU.txt', "a")
            # f.write(str(IoU.item()))
            # f.close()
            break
    
