import logging
import cv2
import mediapipe.python.solutions as mp_solutions

class BaseInferencer():
    def __init__(self, debug_mode: bool=False):
        self.debug_mode: bool = debug_mode
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO)
        self.pose: mp_solutions.pose.Pose = mp_solutions.pose.Pose(static_image_mode=False)
        self.holistic: mp_solutions.holistic.Holistic = mp_solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def inference(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        if results.pose_landmarks:
            mp_solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=results.pose_landmarks,
                connections=mp_solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec = mp_solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec = mp_solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
        if results.left_hand_landmarks:
            mp_solutions.drawing_utils.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_solutions.holistic.HAND_CONNECTIONS,
                mp_solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

        if results.right_hand_landmarks:
            mp_solutions.drawing_utils.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_solutions.holistic.HAND_CONNECTIONS,
                mp_solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

        if results.face_landmarks:
            mp_solutions.drawing_utils.draw_landmarks(
                image,
                results.face_landmarks,
                mp_solutions.holistic.FACEMESH_TESSELATION,
                mp_solutions.drawing_utils.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_solutions.drawing_utils.DrawingSpec(color=(80, 256, 121), thickness=1)
            )
        return image, results.pose_landmarks