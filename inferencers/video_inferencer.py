from typing import Union

import cv2 as cv
import numpy as np

from data_processing.exercise_classifier import classify_exercise_from_angles
from inferencers.base_inferencer import BaseInferencer
from utils import write_joint_angles_csv_row, write_pose_csv_row, write_pose_embedding_csv_row


class VideoInferencer(BaseInferencer):
    """The following class processes an video or camera stream using cv2 and mediapipe
    and returns the saves the video."""

    def __init__(self, debug_mode: bool = False):
        super().__init__(debug_mode=debug_mode, static_image_mode=False)

    def inference(self, stream_path: Union[str, int]=0, output_path: str=None,
                        show=True, should_infer: bool=True, save_csv: str=None):
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

            video_name = str(stream_path) if isinstance(stream_path, str) else "webcam"
            frame_count = 0
            header_written = False

            while cap.isOpened():
                if not ret:
                    self.logger.info("End of video stream.")
                    break

                if should_infer:
                    frame, landmarks = super().inference(frame)
                    if landmarks is not None:
                        lm_array = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks.landmark])
                        embedding = self.embedder(lm_array)
                        features.append(embedding.flatten())
                        write_pose_embedding_csv_row(
                            csv_path="output/pose_embeddings.csv",
                            video_name=video_name,
                            frame_number=frame_count,
                            embedding=embedding,
                            embedding_names=self.embedder.get_embedding_names(),
                            write_header=not header_written
                        )

                        angle_dict = self.angle_calculator(lm_array)
                        print(angle_dict)
                        write_joint_angles_csv_row(
                            csv_path="output/joint_angles.csv",
                            video_name=video_name,
                            frame_number=frame_count,
                            angle_dict=angle_dict,
                            write_header=not header_written
                        )

                        exercise = classify_exercise_from_angles(angle_dict)
                        angle_dict["exercise"] = exercise  # include in CSV
                        self.logger.info(f"Frame {frame_count}: Detected exercise = {exercise}")


                    if save_csv:
                        if not header_written:
                            write_pose_csv_row(save_csv, video_name, frame_count, landmarks, write_header=True)
                            header_written = True
                        else:
                            write_pose_csv_row(save_csv, video_name, frame_count, landmarks)

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
