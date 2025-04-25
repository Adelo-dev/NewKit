import csv
import logging
import os

import cv2
import mediapipe.python.solutions as mp_solutions
import numpy as np
import tqdm
from PIL import Image, ImageDraw

from data_processing.classification_smoothing import EMADictSmoothing
from data_processing.pose_embedding import FullBodyPoseEmbedder


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
        self.embedder = FullBodyPoseEmbedder()
        self.pose_classifier_filter = EMADictSmoothing()
        # self.repetition_counter = RepetitionCounter()
        # self.pose_classifier_visualizer = PoseClassificationVisualizer()
    @staticmethod
    def bgr_to_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def rgb_to_bgr(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
    def flatten_landmark_features(landmarks, frame_width, frame_height):
        features = []
        for lm in landmarks.landmark:
            features.extend([lm.x * frame_width, lm.y * frame_height, lm.z * frame_width])
        return features

    @staticmethod
    def draw_hud(image) -> None:
        BaseInferencer.put_text_safe(image, "Press 'q' to quit", (10, 30))

    def get_class_names(self, images_in_folder: str) -> str:
        pose_class_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])
        return pose_class_names

    def inference(self, image) -> tuple:
        image_rgb = self.bgr_to_rgb(image)
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

    def draw_xz_projection(self, output_frame, pose_landmarks, r=0.5, color='red'):
        frame_height, frame_width = output_frame.shape[:2]
        img = Image.new('RGB', (frame_width, frame_height), color='white')

        if pose_landmarks is None or len(pose_landmarks) != 33:
            return np.asarray(img)

        r *= frame_width * 0.01
        draw = ImageDraw.Draw(img)

        for idx_1, idx_2 in mp_solutions.pose.POSE_CONNECTIONS:
            try:
                x1, y1, z1 = pose_landmarks[idx_1] * [1, 1, -1] + [0, 0, frame_height * 0.5]
                x2, y2, z2 = pose_landmarks[idx_2] * [1, 1, -1] + [0, 0, frame_height * 0.5]
                draw.ellipse([x1 - r, z1 - r, x1 + r, z1 + r], fill=color)
                draw.ellipse([x2 - r, z2 - r, x2 + r, z2 + r], fill=color)
                draw.line([x1, z1, x2, z2], width=int(r), fill=color)
            except IndexError:
                continue

        return np.asarray(img)

    def bootstrap_from_folder(self, images_in_folder, images_out_folder, csvs_out_folder, per_pose_class_limit=None):
        pose_class_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])
        os.makedirs(csvs_out_folder, exist_ok=True)

        for class_name in pose_class_names:
            print(f'Bootstrapping: {class_name}')
            class_in_path = os.path.join(images_in_folder, class_name)
            class_out_path = os.path.join(images_out_folder, class_name)
            class_csv_path = os.path.join(csvs_out_folder, class_name + '.csv')
            os.makedirs(class_out_path, exist_ok=True)

            image_names = sorted([n for n in os.listdir(class_in_path) if not n.startswith('.')])
            if per_pose_class_limit:
                image_names = image_names[:per_pose_class_limit]

            with open(class_csv_path, 'w') as csv_file:
                writer = csv.writer(csv_file)
                for image_name in tqdm.tqdm(image_names):
                    input_path = os.path.join(class_in_path, image_name)
                    image = cv2.imread(input_path)
                    if image is None:
                        continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    frame_height, frame_width = image.shape[:2]

                    with mp_solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose_tracker:
                        result = pose_tracker.process(image=image_rgb)
                        landmarks = result.pose_landmarks

                    output_image = image_rgb.copy()
                    if landmarks is not None:
                        if landmarks:
                            self.draw_landmarks_safe(
                                output_image,
                                landmarks,
                                mp_solutions.pose.POSE_CONNECTIONS,
                                landmark_color=(0, 255, 0),
                                connection_color=(0, 0, 255),
                                thickness=2,
                                radius=2
                            )
                    output_bgr = self.rgb_to_bgr(output_image)
                    output_image_path = os.path.join(class_out_path, image_name)
                    cv2.imwrite(output_image_path, output_bgr)

                    if result.pose_landmarks is not None:
                        flat_landmarks = self.flatten_landmark_features(
                        result.pose_landmarks,
                        frame_width,
                        frame_height)
                        if len(flat_landmarks) == 99:
                            writer.writerow([image_name] + list(map(str, flat_landmarks)))

                        # Also save the XZ projection
                        landmark_array = np.array(flat_landmarks, dtype=np.float32).reshape((33, 3))
                        xz_img = self.draw_xz_projection(output_bgr, landmark_array)
                        combined_img = np.concatenate((output_bgr, xz_img), axis=1)
                        cv2.imwrite(output_image_path, combined_img)

    def align_images_and_csvs(self, images_out_folder, csvs_out_folder, print_removed_items=False):
        pose_classes = sorted([n for n in os.listdir(images_out_folder) if not n.startswith('.')])
        for class_name in pose_classes:
            class_out_folder = os.path.join(images_out_folder, class_name)
            csv_out_path = os.path.join(csvs_out_folder, class_name + '.csv')

            # Read all rows from the CSV file
            rows, image_names_in_csv = [], []
            with open(csv_out_path) as csv_out_file:
                csv_out_reader = csv.reader(csv_out_file, delimiter=',')
                for row in csv_out_reader:
                    rows.append(row)

            # Re-write the CSV only with valid images
            with open(csv_out_path, 'w', newline='') as csv_out_file:
                csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                for row in rows:
                    if not row:
                        continue
                    image_name = row[0]
                    image_path = os.path.join(class_out_folder, image_name)
                    if os.path.exists(image_path):
                        image_names_in_csv.append(image_name)
                        csv_out_writer.writerow(row)
                    elif print_removed_items:
                        print(f'[CSV] Removed missing image from CSV: {image_path}')

            # Remove images that are not in CSV
            for image_name in os.listdir(class_out_folder):
                if image_name not in image_names_in_csv:
                    image_path = os.path.join(class_out_folder, image_name)
                    os.remove(image_path)
                    if print_removed_items:
                        print(f'[Folder] Removed orphaned image: {image_path}')

    def analyze_outliers(self, outliers):
        for outlier in outliers:
            image_path = os.path.join("fitness_poses_images_out", outlier.sample.class_name, outlier.sample.name)
            print(f'[OUTLIER] sample={image_path}')
            print(f'  expected={outlier.sample.class_name}')
            print(f'  predicted={outlier.detected_class}')
            print(f'  all predictions={outlier.all_classes}')
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] Could not load image: {image_path}")
                continue  # Skip to next outlier

            cv2.imshow('Outlier', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def remove_outliers(self, outliers):
        for outlier in outliers:
            path = os.path.join(outlier.sample.class_name, outlier.sample.name)
            if os.path.exists(path):
                os.remove(path)

    def print_images_in_statistics(self, images_folder: str):
        class_names =  self.get_class_names(images_folder)
        self.print_images_statistics(images_folder, class_names)

    def print_images_out_statistics(self, images_folder: str):
        class_names =  self.get_class_names(images_folder)
        self.print_images_statistics(images_folder, class_names)

    def print_images_statistics(self, images_folder: str, pose_class_names: str):
        print('Number of images per pose class:')
        for pose_class_name in pose_class_names:
            n_images = len([n for n in os.listdir(os.path.join(images_folder, pose_class_name))
                if not n.startswith('.')])
            print('  {}: {}'.format(pose_class_name, n_images))
