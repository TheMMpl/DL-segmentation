# DL-segmentation
The aim of this projet is to perform image segmentation using a convolutional neural network.
## Donwloading dataset
For downloading dataset set `DOWNLOAD_DATASET=1` in `./src/consts.py` and provide your username and password for Cityscapes account.
## Running inference
## Installation
Clone the repository and build the docker image with
```
docker build -t segmentation .
```
to run inference from the web interface with wandb use:
```
sudo docker  run -p 5000:5000 -v $(pwd):$(pwd) -w $(pwd) -it -t  segmentation sh -c "wandb login <your_api_key> && flask --app src/dl_segmentation/server/server run --host 0.0.0.0 --debug"
```
### With wandb (old)
If you wish to run the full tain or eval pipelines, wandb access is required. Building the docker image, provide your wandb api key with
```
export WANDB_API_KEY=<your_key_here>
```
First, clone the repository, then in the `DL_segmentation` directory build the docker image using
```
docker build -t segmentation --secret id=WANDB_API_KEY,env=WANDB_API_KEY .
```
run the flask app inside the container with 
```
sudo docker  run -p 5000:5000 -v $(pwd):$(pwd) -w $(pwd) -it -t  segmentation flask --app src/dl_segmentation/server/server run --host 0.0.0.0 --debug
```

```
wandb docker-run -p 5000:5000 -v $(pwd):$(pwd) -w $(pwd) -it -t  segmentation flask --app src/dl_segmentation/server/server run --host 0.0.0.0 --debug
```