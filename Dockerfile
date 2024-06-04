FROM python:3.8
WORKDIR ./
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["flask --app src/dl_segmentation/server/server run --debug"]
