import numpy as np

class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into normalized 3D embedding vectors between body joints."""

    def __init__(self, torso_size_multiplier=2.5):
        self._torso_size_multiplier = torso_size_multiplier

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

        self._embedding_names = []  # Populated during embedding

    def __call__(self, landmarks):
        assert landmarks.shape[0] == len(self._landmark_names), \
            f'Expected {len(self._landmark_names)} landmarks, got {landmarks.shape[0]}'
        
        landmarks = np.copy(landmarks)
        landmarks = self._normalize_pose_landmarks(landmarks)
        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding

    def get_embedding_names(self):
        """Returns list of landmark name pairs for each embedding vector."""
        return self._embedding_names

    def _normalize_pose_landmarks(self, landmarks):
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        landmarks *= 100
        return landmarks

    def _get_pose_center(self, landmarks):
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        return (left_hip + right_hip) * 0.5

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        landmarks_2d = landmarks[:, :2]
        left_hip = landmarks_2d[self._landmark_names.index('left_hip')]
        right_hip = landmarks_2d[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5
        left_shoulder = landmarks_2d[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks_2d[self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5
        torso_size = np.linalg.norm(shoulders - hips)
        pose_center = self._get_pose_center(landmarks_2d)
        max_dist = np.max(np.linalg.norm(landmarks_2d - pose_center, axis=1))
        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        embedding = []
        self._embedding_names = []  # Reset each time

        def add(name_from, name_to):
            embedding.append(self._get_distance_by_names(landmarks, name_from, name_to))
            self._embedding_names.append((name_from, name_to))

        def add_avg_to_avg(label, name_from_1, name_from_2, name_to_1, name_to_2):
            a = self._get_average_by_names(landmarks, name_from_1, name_from_2)
            b = self._get_average_by_names(landmarks, name_to_1, name_to_2)
            embedding.append(self._get_distance(a, b))
            self._embedding_names.append((f"{name_from_1}_{name_from_2}_avg", f"{name_to_1}_{name_to_2}_avg"))

        # Mid-hip to mid-shoulder
        add_avg_to_avg("centerline", "left_hip", "right_hip", "left_shoulder", "right_shoulder")

        # Arms (one joint)
        add("left_shoulder", "left_elbow")
        add("right_shoulder", "right_elbow")
        add("left_elbow", "left_wrist")
        add("right_elbow", "right_wrist")

        # Legs (one joint)
        add("left_hip", "left_knee")
        add("right_hip", "right_knee")
        add("left_knee", "left_ankle")
        add("right_knee", "right_ankle")

        # Arms (two joints)
        add("left_shoulder", "left_wrist")
        add("right_shoulder", "right_wrist")

        # Legs (two joints)
        add("left_hip", "left_ankle")
        add("right_hip", "right_ankle")

        # Arms (four joints)
        add("left_hip", "left_wrist")
        add("right_hip", "right_wrist")

        # Arms (five joints)
        add("left_shoulder", "left_ankle")
        add("right_shoulder", "right_ankle")

        # Cross body
        add("left_elbow", "right_elbow")
        add("left_knee", "right_knee")
        add("left_wrist", "right_wrist")
        add("left_ankle", "right_ankle")

        return np.array(embedding)

    def _get_average_by_names(self, landmarks, name1, name2):
        return (landmarks[self._landmark_names.index(name1)] +
                landmarks[self._landmark_names.index(name2)]) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        return self._get_distance(
            landmarks[self._landmark_names.index(name_from)],
            landmarks[self._landmark_names.index(name_to)]
        )

    def _get_distance(self, point_from, point_to):
        return point_to - point_from
