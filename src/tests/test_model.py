import pytorch_lightning as L
import torch
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from torchvision.io.image import read_image
from dl_segmentation.model.model import model, weights

model.eval()
#tu testowy fragment danych
img= read_image("data/01_raw/istockphoto-1279831403-612x612.jpg")
preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0)
prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
mask = normalized_masks[0, class_to_idx["car"]]
#na razie tyle, potem mozna dodac porownywanie mask wedlug metryki