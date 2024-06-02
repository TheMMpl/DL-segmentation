from consts import UPLOAD_FOLDER, RESULTS_FOLDER
import pytest
import os
from flask import Flask, flash, request, redirect, url_for
from dl_segmentation.server.server import app

# @pytest.fixture(scope="session")
# def app():
#     server=app
#     yield server

@pytest.fixture(scope="session")
def client():
    return app.test_client()


@pytest.fixture(scope="session")
def runner():
    return app.test_cli_runner()

def test_request_example(client):
    response = client.get("/")
    assert b"<h1>Upload new File</h1>" in response.data