import os
import uuid
from typing import Union

import cv2 as cv
import numpy as np

from data_processing.pose_classifier import PoseClassifier
from data_processing.repetition_counter import RepetitionCounter
from inferencers.base_inferencer import BaseInferencer


class VideoInferencer(BaseInferencer):
    """The following class processes an video or camera stream using cv2 and mediapipe
    and returns the saves the video."""

    def __init__(self, debug_mode: bool = False):
        super().__init__(debug_mode=debug_mode, static_image_mode=False)

    def inference(self,
                  stream_path: Union[str, int]=0,
                  output_path: str=None,
                  show=True,
                  should_infer: bool=True,
                  classifier_inputs: str=None):
        should_show = show
        cap = cv.VideoCapture(stream_path)
        if classifier_inputs:
            self.logger.info("Pose classification is enabled.")
            pose_classifier = PoseClassifier(pose_samples_file=classifier_inputs)
            rep_counter = RepetitionCounter("pull-up_down")
        video_writer = None
        features = []

        if cap.isOpened():
            ret, frame = cap.read()
            height, width, _ = frame.shape

            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if not output_path.endswith(".mp4"):
                    output_path = f"{output_path.rstrip('/')}/{uuid.uuid4()}.mp4"
                fps = cap.get(cv.CAP_PROP_FPS)
                fourcc = cv.VideoWriter_fourcc(*"mp4v")
                video_writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            while cap.isOpened():
                if not ret:
                    self.logger.info("End of video stream.")
                    break
                if should_infer:
                    frame, landmarks = super().inference(frame)
                if landmarks is not None:
                    features.append(landmarks.landmark)

                    if classifier_inputs:
                        lm_array = np.array(
                            [[lmk.x * width, lmk.y * height, lmk.z * width] for lmk in landmarks.landmark],
                                dtype=np.float32
                        )
                        assert lm_array.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(landmarks.shape)
                        pose_classification = pose_classifier(lm_array)
                        rep_counter(pose_classification)
                        self.logger.info(
                            f"Pose classification: {pose_classification}, Repetition count: {rep_counter.n_repeats}"
                        )

                if should_show:
                    self.draw_hud(frame)
                    cv.imshow("frame", frame)
                    if cv.waitKey(1) == ord("q"):
                        should_show = False
                        cv.destroyAllWindows()

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
