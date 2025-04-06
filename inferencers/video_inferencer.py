import cv2 as cv
from inferencers.base_inferencer import BaseInferencer

class VideoInferencer(BaseInferencer):
    """ The following class processes an video or camera stream using cv2 and mediapipe and returns the saves the video. """
    def __init__(self, debug_mode: bool=False):
        super().__init__(debug_mode=debug_mode)

    def inference(self, video_path: str, output_path: str, show=True, should_infer: bool=True):
        cap = cv.VideoCapture(video_path)
        processed_frames = [] 

        if cap.isOpened():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("End of video stream.")
                    break

                if should_infer:
                    frame, _ = super().inference(frame)

                processed_frames.append(frame)

                if show:
                    cv.imshow('frame', frame)
                    if cv.waitKey(1) == ord('q'):
                        break

        else:
            self.logger.error("Error: Unable to open video stream.")
            return

        height, width, _ = processed_frames[0].shape
        fps = cap.get(cv.CAP_PROP_FPS)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in processed_frames:
            out.write(frame)

        out.release()
        cap.release()
        cv.destroyAllWindows()

        if out:
            self.logger.info(f"Video saved to {output_path}.")
        else:
            self.logger.error("Error: Unable to save video.")