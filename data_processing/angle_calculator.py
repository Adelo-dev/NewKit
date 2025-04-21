import numpy as np


class JointAngleCalculator:
    """
    Calculates joint angles (in degrees) from normalized 3D pose landmarks.
    """

    def __init__(self):
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

    def __call__(self, landmarks):
        """
        Args:
            landmarks (np.ndarray): NumPy array of shape (33, 3)

        Returns:
            dict: Joint angles in degrees keyed by joint name.
        """
        assert landmarks.shape[0] == 33, f"Expected 33 landmarks, got {landmarks.shape[0]}"
        angles = {}

        def lmk(name):
            return landmarks[self._landmark_names.index(name)]

        def add_angle(joint_name, a, b, c):
            angle = self.calculate_angle(a, b, c)
            angles[joint_name] = round(angle, 2)

        # Upper body
        add_angle("left_elbow",  lmk("left_shoulder"), lmk("left_elbow"), lmk("left_wrist"))
        add_angle("right_elbow", lmk("right_shoulder"), lmk("right_elbow"), lmk("right_wrist"))
        add_angle("left_shoulder",  lmk("left_elbow"), lmk("left_shoulder"), lmk("left_hip"))
        add_angle("right_shoulder", lmk("right_elbow"), lmk("right_shoulder"), lmk("right_hip"))

        # Lower body
        add_angle("left_knee",  lmk("left_hip"), lmk("left_knee"), lmk("left_ankle"))
        add_angle("right_knee", lmk("right_hip"), lmk("right_knee"), lmk("right_ankle"))
        add_angle("left_hip",  lmk("left_shoulder"), lmk("left_hip"), lmk("left_knee"))
        add_angle("right_hip", lmk("right_shoulder"), lmk("right_hip"), lmk("right_knee"))

        return angles

    @staticmethod
    def calculate_angle(a, b, c):
        """
        Returns the angle at point 'b' formed by points 'a' and 'c'.
        """
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle_rad)
