# DL-segmentation
We trained a resudual Unet architecture to perform image segmentation on the Cityscapes Dataset.
Here you can view some sample results 


## Installation
Clone the repository and build the docker image with
```
docker build -t segmentation .
```
## Running inference
to run inference from the web interface with wandb use:
```
sudo docker  run -p 5000:5000 -v $(pwd):$(pwd) -w $(pwd) -it -t  segmentation sh -c "wandb login <your_api_key> && flask --app src/dl_segmentation/server/server run --host 0.0.0.0 --debug"
```
## Running pipelines
To download the dataset set `DOWNLOAD_DATASET=1` in `./src/consts.py` and provide your username and password for Cityscapes account.

```
kedro run --pipeline=data_processing
```

