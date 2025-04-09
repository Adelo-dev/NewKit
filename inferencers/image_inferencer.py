import numpy
import cv2 as cv
from inferencers.base_inferencer import BaseInferencer
class ImageInference(BaseInferencer):
    """Class for performing image inference using MediaPipe Pose.
    This class uses the MediaPipe Pose solution to detect and draw pose landmarks on images.
    It can process images in static mode and save the output to a specified path.
    """

    def __init__(self, debug_mode: bool=False):
        super().__init__(debug_mode=debug_mode)

    def inference(self, image_path: str, output_path: str=None, show=True, should_infer: bool=True) -> None:
        image: numpy.ndarray = cv.imread(image_path)
        pose_landmarks = None
        if image is None:
            self.logger.error(f"Error: Unable to load image at {image_path}.")
            return

        if should_infer:
            image, pose_landmarks = super().inference(image=image)
            self.logger.info("Finished processing the image.")
            self.logger.debug(f"Number of landmarks detected: {len(pose_landmarks.landmark)}")

        if output_path:
            cv.imwrite(output_path, image)
            self.logger.info(f"Output image saved to {output_path}.")

        if show:
            cv.imshow('frame', image)
            cv.waitKey(0)
        
        return pose_landmarks
