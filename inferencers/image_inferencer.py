import cv2 as cv
import numpy

from inferencers.base_inferencer import BaseInferencer
from utils import write_pose_csv_row


class ImageInference(BaseInferencer):
    """Detects and annotates pose landmarks in images using MediaPipe Pose."""

    def __init__(self, debug_mode: bool = False):
        super().__init__(debug_mode=debug_mode)

    def inference(self, image_path: str,
                        output_path: str=None,
                        show=True,
                        should_infer: bool=True,
                        save_csv: str=None):
        image: numpy.ndarray = cv.imread(image_path)
        pose_landmarks = None
        frame_number = 0
        if image is None:
            self.logger.error(f"Error: Unable to load image at {image_path}.")
            return

        if should_infer:
            image, pose_landmarks = super().inference(image=image)
            self.logger.info("Finished processing the image.")
            self.logger.debug(
                f"Number of landmarks detected: {len(pose_landmarks.landmark)}"
            )
            if pose_landmarks:
                self.logger.debug(
                    f"Number of landmarks detected: {len(pose_landmarks.landmark)}"
                )

                if save_csv:
                    write_pose_csv_row(
                        csv_path=save_csv,
                        video_name=str(image_path),
                        frame_number=frame_number,
                        landmark_data=pose_landmarks,
                        write_header=True  # Only one image
                    )

        if output_path:
            cv.imwrite(output_path, image)
            self.logger.info(f"Output image saved to {output_path}.")

        if show:
            self.draw_hud(image)
            cv.imshow("frame", image)
            cv.waitKey(0)

        return pose_landmarks
