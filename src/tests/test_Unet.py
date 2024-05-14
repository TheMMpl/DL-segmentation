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

NUM_CLASSES=34
data_params=("demo_results/test_test",'./dataset/cityscapes')
model_checkpoint='dlprojekt/DL_segmenation/model-daeah9nu:v37'
test_size=10
metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES)

#@pytest.mark.parametrize("results_path,dataset_path",data_params)
@pytest.fixture()
def prepare_data(request):
    Path(request.param[0]).mkdir(parents=True, exist_ok=True)
    val_dataset = Cityscapes(request.param[1], split='val',target_type='semantic', transforms = CityScapesTransform())
    return val_dataset, request.param[0]


#@pytest.mark.parametrize("model_checkpoint",model_checkpoint)
@pytest.fixture()
def prepare_model(request):
    run = wandb.init(project="DL_segmenation")
    artifact = run.use_artifact(request.param, type="model")
    artifact_dir = artifact.download()
    print(artifact_dir)
    model = LightningModel.load_from_checkpoint(Path(artifact_dir) / "model.ckpt",num_classes=34)
    model.eval()
    model.unet.eval()
    return model
 

@pytest.mark.parametrize("test_size,metric,prepare_data,prepare_model",[(test_size,metric,data_params,model_checkpoint)],indirect=["prepare_data","prepare_model"])
def test_model(test_size,metric,prepare_data,prepare_model):
    model=prepare_model
    val_dataset,results_path=prepare_data
    print(val_dataset)
    print(test_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trans = [transforms.v2.ToDtype(torch.float32, scale=True), transforms.v2.Resize((256,512))]
    image_num=0
    for img,target in val_dataset:
        image_num+=1
        for t in trans:
            img=t(img)
        image=img.to(device)
        image.unsqueeze_(0)
        result=torch.argmax(model(image)[0],dim=0).cpu().detach().numpy()
        comparison=np.vstack([result,target[0]])
        #plt.imsave(f'demo_results/overfit_test37/res{jank_iter}.jpg',comparison)
        if image_num>test_size:
            break
    

#def test_Unet(model_path,results_path,dataset_path,test_size,metric):

#trainer = L.Trainer(limit_train_batches=2, max_epochs=1)
#train_loader=DataLoader(Cityscapes)