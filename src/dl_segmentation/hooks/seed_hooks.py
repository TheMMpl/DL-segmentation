import random
import torch
import numpy as np
import logging
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog


class SettingSeedHooks:

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog) -> None:
        seed = catalog.load("params:seed")
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        logging.getLogger(__name__).info("Set " + str(seed) + " seed.")
        

