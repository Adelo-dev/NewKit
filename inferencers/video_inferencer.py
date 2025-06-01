import os
import uuid
from typing import Tuple, Union

import cv2 as cv
import numpy as np

from data_processing.error_classification import display_class_name
from data_processing.pose_classifier import PoseClassifier, collect_and_classify_pose_images
from data_processing.pose_frame_identifier import extract_best_up_down_frames_from_folder
from data_processing.repetition_counter import RepetitionCounter
from inferencers.base_inferencer import BaseInferencer
from utils.utils import extract_frames_from_videos


class VideoInferencer(BaseInferencer):
    """The following class processes an video or camera stream using cv2 and mediapipe
    and returns the saves the video."""

    def __init__(self, debug_mode: bool = False):
        super().__init__(debug_mode=debug_mode, static_image_mode=False)

    def inference(self,
                  stream_path: Union[str, int]=0,
                  trainer_videos: Union[str, int]=0,
                  output_path: str=None,
                  show=True,
                  should_infer: bool= True,
                  add_new_data= False,
                  classifier_errors= "",
                  exercise_name= "" ,
                  classifier_rep_count= "") -> Tuple[list, list]:
        should_show = show
        cap = cv.VideoCapture(stream_path)

        video_writer = None
        features = []
        classifier_prediction = []
        if add_new_data:
            extract_frames_from_videos(trainer_videos, "frames", exercise_name=exercise_name)
            print(trainer_videos)
            extract_best_up_down_frames_from_folder(
                videos_folder= f"{trainer_videos}/good_form",
                output_dir= "frames",
                exercise_name= exercise_name,
                top_k= 40
            )
            collect_and_classify_pose_images(base_dir="frames", exercise_name=exercise_name)
        if classifier_rep_count or classifier_errors:
            self.logger.info("Pose classification is enabled.")
            pose_classifier_reps_count = PoseClassifier(pose_samples_file=classifier_rep_count)
            pose_classifier_errors = PoseClassifier(pose_samples_file=classifier_errors)
            rep_counter = RepetitionCounter(exercise_name +"_down")

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

                    if classifier_rep_count or classifier_errors:
                        lm_array = np.array(
                            [[lmk.x * width, lmk.y * height, lmk.z * width] for lmk in landmarks.landmark],
                                dtype=np.float32
                        )
                        assert lm_array.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(landmarks.shape)
                        print(lm_array.shape)
                        if classifier_rep_count:
                            pose_classification_reps = pose_classifier_reps_count(lm_array)
                            print("Incoming landmark array:", lm_array)
                            print(pose_classification_reps)
                            rep_counter(pose_classification_reps)
                            classifier_prediction.append(max(pose_classification_reps))
                            self.put_text_safe(frame, f"Pose: {max(pose_classification_reps)}", (10, 60))
                            self.put_text_safe(frame, f"Reps: {rep_counter.n_repeats}", (10, 90))
                        if classifier_errors:
                            pose_classification_errors = pose_classifier_errors(lm_array)
                            if pose_classification_errors:
                                # Get the top class and its vote count
                                top_class = max(pose_classification_errors, key=pose_classification_errors.get)
                                top_count = pose_classification_errors[top_class]

                                if top_count >= 9:
                                    self.put_text_safe(
                                        frame,
                                        f"Mistakes: {display_class_name(top_class)}",
                                        position=(10, 120),
                                        color=(0, 0, 255)
                                    )
                                    #self.logger.debug(f"'{top_class}' with count {top_count}")
                                else:
                                    #self.logger.debug(f"'{top_class}' with count {top_count} test test")
                                    self.put_text_safe(
                                        frame,
                                        f"Mistakes: {display_class_name(top_class)}",
                                        position=(10, 120),
                                        color=(0, 0, 255)
                                    )
                            else:
                                self.logger.warning("No error classification result available for this frame.")

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
            return np.array(features), classifier_prediction
        else:
            self.logger.error("Error: Unable to open video stream.")
            return
