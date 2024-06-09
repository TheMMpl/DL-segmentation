FROM python:3.8
WORKDIR ./
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python3 -m build
# RUN --mount=type=secret,id=WANDB_API_KEY \
# export WANDB_API_KEY=$(cat /run/secrets/WANDB_API_KEY) \
# wandb login
EXPOSE 5000
# CMD ["flask --app src/dl_segmentation/server/server run --debug"]
