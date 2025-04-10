import os
import uuid
import pytest
from inferencers.base_inferencer import BaseInferencer
from inferencers.image_inferencer import ImageInference
from inferencers.video_inferencer import VideoInferencer

@pytest.fixture
def base_inferencer():
    return BaseInferencer()

@pytest.fixture
def image_inferencer():
    return ImageInference()

@pytest.fixture
def video_inferencer():
    return VideoInferencer()

def test_base_inferencer_initialization(base_inferencer):
    assert base_inferencer is not None

def test_image_inferencer_initialization(image_inferencer):
    assert image_inferencer is not None

def test_video_inferencer_initialization(video_inferencer):
    assert video_inferencer is not None

def test_image_inferencer_specific_method(image_inferencer):
    output_path = f'output/{uuid.uuid4()}.jpg'
    result = image_inferencer.inference(image_path="sample_data/sample.jpg",
                                        output_path=output_path,
                                        show=False)
    assert os.path.exists(output_path) and os.path.getsize(output_path) > 0
    assert result is not None
    os.remove(output_path)

def test_video_inferencer_specific_method(video_inferencer):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', 'output', f'{uuid.uuid4()}.mp4')
    print("CWD:", os.getcwd())
    video_inferencer.inference(stream_path="sample_data/sample_dips.mp4",
                                        output_path=output_path,
                                        show=False)
    assert os.path.exists(output_path) and os.path.getsize(output_path) > 0
    os.remove(output_path)
