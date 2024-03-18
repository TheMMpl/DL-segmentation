from model.model import model
import pytorch_lightning as L
from model.model import LightningModel
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

unet=LightningModel(20)

wandb_logger=WandbLogger(project="DL_segmenation",log_model="all")
checkpoint_callback=ModelCheckpoint(monitor="val_loss",save_top_k=10,every_n_epochs=1)
trainer=Trainer(logger=wandb_logger,callbacks=[checkpoint_callback])

#to complete with correct dataloader
train_loader=DataLoader(Cityscapes)

trainer.fit(model=unet,train_dataloaders=train_loader)


#some loading code for later

# reference can be retrieved in artifacts panel
# "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"

# download checkpoint locally (if not already cached)
#wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
# load checkpoint
#model = LightningModel.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")


