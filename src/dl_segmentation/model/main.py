import random
import torch
import numpy as np

from dl_segmentation.model.consts import MODEL_SEED

random.seed(MODEL_SEED)
torch.manual_seed(MODEL_SEED)
np.random.seed(MODEL_SEED)

from dl_segmentation.model.random_seed_test import print_random_ints

print_random_ints(10)
print(random.randint(1,100))
