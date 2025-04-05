import cv2
import mediapipe as mp
class ImageInference():
    """ The following class processes an image using cv2 and mediapipe and returns the picture"""
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def inference(self, image_path: str, output_path: str, should_infer: bool=False):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if should_infer:
            results = self.pose.process(image_rgb)
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )     
                cv2.imwrite(output_path, image)
                return
        cv2.imwrite(output_path, image)
