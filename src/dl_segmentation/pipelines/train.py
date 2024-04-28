import pytorch_lightning as L
from dl_segmentation.model.model import LightningModel
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision import transforms
from torchvision.transforms import v2
import torch

class CityScapesTransform:
    def __call__(self, image, target):
        trans_im = [transforms.PILToTensor(), transforms.v2.ToDtype(torch.float32, scale=True), transforms.v2.Resize((128,256))]
        trans_tar = [transforms.PILToTensor(), transforms.v2.ToDtype(torch.long, scale=False), transforms.v2.Resize((128,256),interpolation=transforms.InterpolationMode.NEAREST)]
        for t in trans_im:
            image = t(image)
        for t in trans_tar:
            target = t(target)
        return image, target



# To wszystko co tu było jest w data_science.nodes.py

#some loading code for later

# reference can be retrieved in artifacts panel
# "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"

# download checkpoint locally (if not already cached)
#wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
# load checkpoint
#model = LightningModel.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")


