import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_processing.classification_smoothing import EMADictSmoothing
from data_processing.pose_embedding import FullBodyPoseEmbedder
from data_processing.pose_sample import PoseSample
from data_processing.utils import generate_pose_samples_from_images


class PoseClassifier(object):
  """Classifies pose landmarks."""

  def __init__(self,
               pose_samples_file=None,
               n_landmarks=33,
               n_dimensions=3,
               top_n_by_max_distance=30,
               top_n_by_mean_distance=10,
               axes_weights=(1., 1., 0.2)):
    self._logger = logging.getLogger(name=self.__class__.__name__)
    self._pose_embedder = FullBodyPoseEmbedder()
    self._pose_classifier_filter = EMADictSmoothing()
    self._n_landmarks = n_landmarks
    self._n_dimensions = n_dimensions
    self._top_n_by_max_distance = top_n_by_max_distance
    self._top_n_by_mean_distance = top_n_by_mean_distance
    self._axes_weights = axes_weights
    self._pose_samples: List[PoseSample] = self.import_pose_samples_from_csv(pose_samples_file)

  def import_pose_samples_from_csv(self, pose_samples_file):
      if not pose_samples_file or not os.path.exists(pose_samples_file):
        self._logger.warning(f"Pose samples file '{pose_samples_file}' does not exist.")
        return []
      pose_samples_df = pd.read_csv(pose_samples_file)
      pose_samples_df['landmarks'] = pose_samples_df['landmarks'].apply(lambda x: np.array(json.loads(x),
                                                                                          np.float32).reshape([self._n_landmarks,
                                                                                                               self._n_dimensions]))
      pose_samples_df['embeddings'] = pose_samples_df['landmarks'].apply(self._pose_embedder)
      return [PoseSample.from_row(row) for _, row in pose_samples_df.iterrows()]

  def export_pose_samples_to_csv(self, output_folder):
    """Exports pose samples to the given folder."""
    os.makedirs(output_folder, exist_ok=True)
    pose_samples_csv_path = os.path.join(output_folder, f'{uuid.uuid4().hex}.csv')
    pd.DataFrame([pose.to_dict() for pose in self._pose_samples]).to_csv(pose_samples_csv_path, index=False)
    return pose_samples_csv_path

  def __call__(self, pose_landmarks):
    """Classifies given pose.

    Classification is done in two stages:
      * First we pick top-N samples by MAX distance. It allows to remove samples
        that are almost the same as given pose, but has few joints bent in the
        other direction.
      * Then we pick top-N samples by MEAN distance. After outliers are removed
        on a previous step, we can pick samples that are closes on average.

    Args:
      pose_landmarks: NumPy array with 3D landmarks of shape (N, 3).

    Returns:
      Dictionary with count of nearest pose samples from the database. Sample:
        {
          'pushups_down': 8,
          'pushups_up': 2,
        }
    """
    # Check that provided and target poses have the same shape.
    assert pose_landmarks.shape == (
      self._n_landmarks,
      self._n_dimensions), \
      'Unexpected shape: {}'.format(pose_landmarks.shape
    )

    # Get given pose embedding.
    pose_embedding = self._pose_embedder(pose_landmarks)
    flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

    # Filter by max distance.
    #
    # That helps to remove outliers - poses that are almost the same as the
    # given one, but has one joint bent into another direction and actually
    # represnt a different pose class.
    max_dist_heap = []
    for sample_idx, sample in enumerate(self._pose_samples):
        if sample.embedding.shape != pose_embedding.shape:
            self._logger.warning(f"Shape mismatch at sample {sample.name}: \
              {sample.embedding.shape} vs {pose_embedding.shape}")
            continue

        max_dist = min(
            np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
            np.max(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
        )
        max_dist_heap.append([max_dist, sample_idx])


    # Filter by mean distance.
    #
    # After removing outliers we can find the nearest pose by mean distance.
    mean_dist_heap = []
    for _, sample_idx in max_dist_heap:
      sample: PoseSample = self._pose_samples[sample_idx]
      mean_dist = min(
          np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
          np.mean(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
      )
      mean_dist_heap.append([mean_dist, sample_idx])

    mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
    mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

    # Collect results into map: (class_name -> n_samples)
    class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
    return {class_name: class_names.count(class_name) for class_name in set(class_names)}

  def analyze_and_remove_outliers(self):
    """Analyzes and removes outliers from the pose samples."""
    cleaned_pose_samples = []
    for sample in tqdm(self._pose_samples, desc='Analyzing pose samples'):
      pose_landmarks = sample.landmarks.copy()
      pose_classification = self.__call__(pose_landmarks)
      class_names = [class_name for class_name, \
        count in pose_classification.items() if count == max(pose_classification.values())]

      # Sample is an outlier if nearest poses have different class or more than
      # one pose class is detected as nearest.
      if sample.class_name in class_names or len(class_names) == 1:
        cleaned_pose_samples.append(sample)
    return cleaned_pose_samples

  def generate_pose_samples(self, images_input_folder, output_folder):
    """Generates pose samples from images in the given folder."""
    self.print_images_statistics(images_input_folder)
    self._pose_samples = generate_pose_samples_from_images(images_input_folder=images_input_folder,
                                      landmarks_shape=(self._n_landmarks, self._n_dimensions))
    self._pose_samples = self.analyze_and_remove_outliers()
    return self.export_pose_samples_to_csv(output_folder)

  def print_images_statistics(self, images_folder: str):
    pose_class_names = sorted([n for n in os.listdir(images_folder) if not n.startswith('.')])
    self._logger.info('Pose classes: {}'.format(pose_class_names))
    for pose_class_name in pose_class_names:
        n_images = len([n for n in os.listdir(os.path.join(images_folder, pose_class_name))
            if not n.startswith('.')])
        self._logger.info('  {}: {}'.format(pose_class_name, n_images))

def collect_and_classify_pose_images(base_dir, exercise_name):
    temp_dir_errors = tempfile.mkdtemp()
    temp_dir_reps = tempfile.mkdtemp()
    output_dir = Path(f"data/exercises/{exercise_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define folders
    folders = {
        "bad_form": Path(base_dir) / exercise_name / "bad_form",
        f"{exercise_name}_up": Path(base_dir) / exercise_name / "good_form" / f"{exercise_name}_up",
        f"{exercise_name}_down": Path(base_dir) / exercise_name / "good_form" / f"{exercise_name}_down",
    }

    # Copy ALL into errors temp dir
    for class_name, src in folders.items():
        if src.is_dir():
            dst = Path(temp_dir_errors) / class_name
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.[jpJP][pnPN]*[gG]"):
                shutil.copy(f, dst / f.name)

    # Copy only good_form into reps temp dir
    for key in [f"{exercise_name}_up", f"{exercise_name}_down"]:
        src = folders[key]
        if src.is_dir():
            dst = Path(temp_dir_reps) / key
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.[jpJP][pnPN]*[gG]"):
                shutil.copy(f, dst / f.name)

    # Generate CSVs using PoseClassifier
    classifier = PoseClassifier()

    csv_path_errors = Path(classifier.generate_pose_samples(temp_dir_errors, str(output_dir)))
    final_path_errors = output_dir / f"{exercise_name}_errors.csv"

    csv_path_reps = Path(classifier.generate_pose_samples(temp_dir_reps, str(output_dir)))
    final_path_reps = output_dir / f"{exercise_name}_rep_count.csv"

    # Append or move errors.csv
    if final_path_errors.exists():
        df = pd.concat([pd.read_csv(final_path_errors), pd.read_csv(csv_path_errors)], ignore_index=True)
        df.to_csv(final_path_errors, index=False)
        csv_path_errors.unlink()
    else:
        csv_path_errors.rename(final_path_errors)

    # Append or move rep_count.csv
    if final_path_reps.exists():
        df = pd.concat([pd.read_csv(final_path_reps), pd.read_csv(csv_path_reps)], ignore_index=True)
        df.to_csv(final_path_reps, index=False)
        csv_path_reps.unlink()
    else:
        csv_path_reps.rename(final_path_reps)

    print(f"✅ Errors CSV saved to: {final_path_errors}")
    print(f"✅ Rep Count CSV saved to: {final_path_reps}")

