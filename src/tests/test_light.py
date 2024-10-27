from dl_segmentation.pipelines.reporting.nodes import load_model, prepare_data, run_inference
from consts import NUM_CLASSES, MODEL_CHECKPOINT, UPLOAD_FOLDER, RESULTS_FOLDER, DEMO_FOLDER
from pathlib import Path
import pytest
import os

@pytest.fixture(scope="session")
def data():
    images=[]
    for i in range(10):
        images.append(prepare_data([DEMO_FOLDER+f'/img{i}.jpg'])[0])
    return images

@pytest.fixture(scope="session")
def model():
    model=load_model(MODEL_CHECKPOINT)
    return model

def test_inference(model,data):
    results,IoU=run_inference(model,data)
    assert IoU==[]

