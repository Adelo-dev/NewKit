import logging

import cv2
import mediapipe.python.solutions as mp_solutions


class BaseInferencer:
    def __init__(self, debug_mode: bool = False, static_image_mode: bool = True):
        self.debug_mode: bool = debug_mode
        self.logger = logging.getLogger(name=self.__class__.__name__)
        logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO)
        self.pose: mp_solutions.pose.Pose = mp_solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1
        )
        self.holistic: mp_solutions.holistic.Holistic = mp_solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def draw_landmarks_safe(image, landmarks, connections, landmark_color,
                                connection_color, thickness=2, radius=2) -> None:
        if landmarks:
            drawing_utils = mp_solutions.drawing_utils
            drawing_utils.draw_landmarks(
                image=image,
                landmark_list=landmarks,
                connections=connections,
                landmark_drawing_spec=drawing_utils.DrawingSpec(color=landmark_color,
                                                                thickness=thickness,
                                                                circle_radius=radius),
                connection_drawing_spec=drawing_utils.DrawingSpec(color=connection_color,
                                                                thickness=thickness)
            )

    @staticmethod
    def put_text_safe(image, text, position,
                      color=(0, 255, 0), font_scale=1, thickness=2):
        if image is not None:
            cv2.putText(image,
                        text,
                        position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        thickness)

    @staticmethod
    def flatten_landmark_features(landmarks):
    # Flatten into a single feature vector (x, y only, normalized)
        features = []
        for lm in landmarks.landmark:
            features.extend([lm.x, lm.y])
        return features

    @staticmethod
    def draw_hud(image) -> None:
        BaseInferencer.put_text_safe(image, "Press 'q' to quit", (10, 30))

    def inference(self, image) -> tuple:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        landmark_groups = [
            (image, results.pose_landmarks,
            mp_solutions.pose.POSE_CONNECTIONS, (0, 255, 0), (0, 0, 255)),
            (image, results.left_hand_landmarks,
            mp_solutions.holistic.HAND_CONNECTIONS, (255, 0, 0), (255, 255, 255)),
            (image, results.right_hand_landmarks,
            mp_solutions.holistic.HAND_CONNECTIONS, (0, 0, 255), (255, 255, 255)),
            (image, results.face_landmarks,
            mp_solutions.holistic.FACEMESH_TESSELATION, (80, 110, 10), (80, 256, 121),
            1, 1)
        ]

        for landmark in landmark_groups:
            self.draw_landmarks_safe(*landmark)

        return image, results.pose_landmarks
