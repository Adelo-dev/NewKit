import cv2 as cv
import mediapipe as mp

class VideoInferencer():
    """ The following class processes an video or camera stream using cv2 and mediapipe and returns the saves the video. """
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def inference(self, video_path: str, output_path: str, should_infer: bool=False):
        cap = cv.VideoCapture(video_path)
        processed_frames = [] 

        if cap.isOpened():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break  
                if should_infer:
                    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    results = self.pose.process(image_rgb)

                    # Draw landmarks directly on the frame
                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                        )
                    processed_frames.append(frame)
                    cv.imshow('frame', frame)
                    if cv.waitKey(1) == ord('q'):
                        break
                else:
                    processed_frames.append(frame)
                    cv.imshow('frame', frame)
                    if cv.waitKey(1) == ord('q'):
                        break

        else:
            print("Cannot open camera")
            exit()



        height, width, _= processed_frames[0].shape
        fps = cap.get(cv.CAP_PROP_FPS)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in processed_frames:
            out.write(frame)
        out.release()
        cap.release()
        cv.destroyAllWindows()

        if out:
            print(f"Video saved to {output_path}")
        else:
            print("Cannot open video file.")
            exit()
