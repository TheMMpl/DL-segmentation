from typing import Dict, Tuple
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
import pandas as pd
import torch
import os
from dl_segmentation.pipelines.train import CityScapesTransform
from dl_segmentation.pipelines.train import ExtraLabelsTransform
from torchvision import transforms

from consts import BATCH_SIZE, NUM_WORKERS, NUM_CLASSES, CITYSCAPES_PASSWORD, CITYSCAPES_USERNAME, DOWNLOAD_DATASET

def _is_true(x: pd.Series) -> pd.Series:

    return x == "t"


def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x


def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def preprocess_companies(companies: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights=DeepLabV3_ResNet101_Weights.DEFAULT
    model=deeplabv3_resnet101(weights=weights).to(device)
    
    if DOWNLOAD_DATASET == 1:
    	os.system("wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username="+CITYSCAPES_USERNAME+"&password="+CITYSCAPES_PASSWORD+"&submit=Login' https://www.cityscapes-dataset.com/login/; history -d $((HISTCMD-1))")
    	os.system("wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1")
    	os.system("wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3")
    	os.system("unzip gtFine_trainvaltest.zip -d ./dataset/cityscapes")
    	os.system("rm ./dataset/cityscapes/license.txt")
    	os.system("rm ./dataset/cityscapes/README")
    	os.system("unzip leftImg8bit_trainvaltest.zip -d ./dataset/cityscapes")
    	os.system("rm ./dataset/cityscapes/license.txt")
    	os.system("rm ./dataset/cityscapes/README")
    	os.system("rm ./leftImg8bit_trainvaltest.zip")
    	os.system("rm ./gtFine_trainvaltest.zip")

    test_dataset = Cityscapes('./dataset/cityscapes', split='test',
                     target_type='semantic', transforms = ExtraLabelsTransform())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model.eval()
    jank_iter=0
    #we don't vectorize because we need to save the results anyways
    for img,target in test_dataset:
        jank_iter+=1
        with torch.no_grad():
            img=img.to(device).unsqueeze(0)
            result=model(img)
            print(result['out'].shape)
        #result=torch.argmax(model(image)[0],dim=0).cpu().detach().numpy()
        #comparison=np.vstack([result,target[0]])
        #plt.imsave(f'demo_results/overfit_test37/train_res{jank_iter}.jpg',comparison)
        #plt.imsave(f'demo_results/overfit_test2/img{jank_iter}.jpg',torch.permute(img,(1,2,0)).numpy()/255)    
        if jank_iter>10:
            break

    companies["iata_approved"] = _is_true(companies["iata_approved"])
    companies["company_rating"] = _parse_percentage(companies["company_rating"])
    return companies, {"columns": companies.columns.tolist(), "data_type": "companies"}


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
    shuttles["price"] = _parse_money(shuttles["price"])
    return shuttles


def create_model_input_table(
    shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
    rated_shuttles = rated_shuttles.drop("id", axis=1)
    model_input_table = rated_shuttles.merge(
        companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.dropna()
    return model_input_table
