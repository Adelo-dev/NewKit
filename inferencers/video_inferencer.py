import os
from typing import Union

import cv2 as cv
import numpy as np

from data_processing.classification_smoothing import EMADictSmoothing
from data_processing.pose_classifier import PoseClassifier
from data_processing.repetition_counter import RepetitionCounter
from inferencers.base_inferencer import BaseInferencer


class VideoInferencer(BaseInferencer):
    """The following class processes an video or camera stream using cv2 and mediapipe
    and returns the saves the video."""

    def __init__(self, debug_mode: bool = False):
        super().__init__(debug_mode=debug_mode, static_image_mode=False)

    def inference(self, stream_path: Union[str, int]=0, output_path: str=None,
                        show=True, should_infer: bool=True, save_csv: str=None):

        video_name = os.path.splitext(os.path.basename(str(stream_path)))[0] if isinstance(stream_path, str) else "webcam"
        cap = cv.VideoCapture(stream_path)
        video_writer = None
        features = []

        if cap.isOpened():
            ret, frame = cap.read()

            if output_path:
                height, width, _ = frame.shape
                fps = cap.get(cv.CAP_PROP_FPS)

                fourcc = cv.VideoWriter_fourcc(*"mp4v")
                video_writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            images_in_folder = "fitness_poses_images_in"
            images_out_folder = "fitness_poses_images_out"
            self.bootstrap_from_folder(
                images_in_folder= images_in_folder,
                images_out_folder= images_out_folder,
                csvs_out_folder= save_csv
            )
            self.print_images_in_statistics(images_in_folder)
            self.print_images_out_statistics(images_in_folder)
            self.align_images_and_csvs(
                images_out_folder=images_out_folder,
                csvs_out_folder=save_csv,
                print_removed_items=False
            )
            self.print_images_out_statistics(images_in_folder)

            pose_embedder = self.embedder
            pose_classifier = PoseClassifier(
                pose_samples_folder="fitness_poses_csvs_out",
                pose_embedder=pose_embedder,
                top_n_by_max_distance=30,
                top_n_by_mean_distance=10
                )
            # outliers = pose_classifier.find_pose_sample_outliers()
            # self.analyze_outliers(outliers)
            # self.remove_outliers(outliers)
            # self.align_images_and_csvs(
            #     images_out_folder=images_out_folder,
            #     csvs_out_folder=save_csv,
            #     print_removed_items=False
            # )
            self.print_images_out_statistics(images_in_folder)
            rep_counter = RepetitionCounter(class_name=video_name)
            pose_classification_filter = EMADictSmoothing(window_size=10, alpha=0.2)

            while cap.isOpened():
                if not ret:
                    self.logger.info("End of video stream.")
                    break
                if should_infer:
                    frame, landmarks = super().inference(frame)
                    if landmarks is not None:
                        lm_array = np.array([[lmk.x * width, lmk.y * height, lmk.z * width] for lmk in landmarks.landmark], dtype=np.float32)
                        assert lm_array.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(landmarks.shape)
                        pose_classification = pose_classifier(lm_array)
                        rep_counter(pose_classification)
                        print(video_name)
                        print(pose_classification)
                        print(rep_counter.n_repeats)

                if show:
                    self.draw_hud(frame)
                    cv.imshow("frame", frame)
                    if cv.waitKey(1) == ord("q"):
                        break

                if video_writer:
                    video_writer.write(frame)

                ret, frame = cap.read()
                frame_count += 1

            if video_writer:
                self.logger.info(f"Video saved to {output_path}.")
                video_writer.release()

            cap.release()
            cv.destroyAllWindows()
            return np.array(features)
        else:
            self.logger.error("Error: Unable to open video stream.")
            return
