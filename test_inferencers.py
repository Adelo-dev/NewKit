import os
import uuid
import pytest
from inferencers.base_inferencer import BaseInferencer
from inferencers.image_inferencer import ImageInference
from inferencers.video_inferencer import VideoInferencer

@pytest.fixture
def base_inferencer() -> BaseInferencer:
    return BaseInferencer()

@pytest.fixture
def image_inferencer() -> ImageInference:
    return ImageInference()

@pytest.fixture
def video_inferencer() -> VideoInferencer:
    return VideoInferencer()

def test_base_inferencer_initialization(base_inferencer) -> None:
    assert base_inferencer is not None

def test_image_inferencer_initialization(image_inferencer) -> None:
    assert image_inferencer is not None

def test_video_inferencer_initialization(video_inferencer) -> None:
    assert video_inferencer is not None

def test_image_inferencer_specific_method(image_inferencer) -> None:
    output_path = f'output/{uuid.uuid4()}.jpg'
    result = image_inferencer.inference(image_path="sample_data/sample.jpg",
                                        output_path=output_path,
                                        show=False)
    assert os.path.exists(path=output_path)
    assert os.path.getsize(filename=output_path) > 0
    assert result is not None
    os.remove(path=output_path)

def test_video_inferencer_specific_method(video_inferencer) -> None:
    output_path = f'output/{uuid.uuid4()}.mp4'

    video_inferencer.inference(stream_path="sample_data/sample_knuckle_pushups.mp4",
                                        output_path=output_path,
                                        show=False)
    assert os.path.exists(path=output_path)
    assert os.path.getsize(filename=output_path) > 0
    os.remove(path=output_path)
