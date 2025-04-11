import cv2 as cv
from typing import Union
from inferencers.base_inferencer import BaseInferencer


class VideoInferencer(BaseInferencer):
    """The following class processes an video or camera stream using cv2 and mediapipe
    and returns the saves the video."""

    def __init__(self, debug_mode: bool = False):
        super().__init__(debug_mode=debug_mode, static_image_mode=False)

    def inference(self, stream_path: Union[str, int]=0, output_path: str=None,
                        show=True, should_infer: bool=True):
        cap = cv.VideoCapture(stream_path)
        video_writer = None

        if cap.isOpened():
            ret, frame = cap.read()

            if output_path:
                height, width, _ = frame.shape
                fps = cap.get(cv.CAP_PROP_FPS)

                fourcc = cv.VideoWriter_fourcc(*"mp4v")
                video_writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                if not ret:
                    self.logger.info("End of video stream.")
                    break

                if should_infer:
                    frame, _ = super().inference(frame)

                if show:
                    self.draw_hud(frame)
                    cv.imshow("frame", frame)
                    if cv.waitKey(1) == ord("q"):
                        break

                if video_writer:
                    video_writer.write(frame)

                ret, frame = cap.read()

            if video_writer:
                self.logger.info(f"Video saved to {output_path}.")
                video_writer.release()
            else:
                self.logger.error("Error: Unable to save video.")
            cap.release()
            cv.destroyAllWindows()
        else:
            self.logger.error("Error: Unable to open video stream.")
            return
