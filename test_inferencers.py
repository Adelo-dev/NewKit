import os
import uuid

import numpy as np
import pytest

from data_processing.pose_classifier import PoseClassifier
from data_processing.pose_embedding import FullBodyPoseEmbedder
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

    result = video_inferencer.inference(stream_path="testing_video/pulluptest.mov",
                                        output_path=output_path,
                                        show=False,
                                        save_csv="fitness_poses_csvs_out")
    assert os.path.exists(path=output_path)
    assert os.path.getsize(filename=output_path) > 0
    assert result is not None
    os.remove(path=output_path)

def test_video_inferencer_classification(video_inferencer) -> None:
    embedder = FullBodyPoseEmbedder()
    pose_classifier = PoseClassifier(
        pose_samples_folder="fitness_poses_csvs_out",
        pose_embedder=embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10
    )
    pose_classifier(np.ones((33,3)))
    result = (pose_classifier)
    assert isinstance(result, PoseClassifier)

