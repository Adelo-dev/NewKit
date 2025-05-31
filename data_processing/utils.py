
import csv
import os
import re

import cv2
import mediapipe.python.solutions as mp_solutions
import numpy as np
import tqdm
from PIL import Image, ImageDraw

from data_processing.pose_embedding import FullBodyPoseEmbedder
from data_processing.pose_sample import PoseSample
from utils.utils import rgb_to_bgr


def draw_xz_projection(output_frame, pose_landmarks, r=0.5, color='red'):
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

def generate_pose_samples_from_images(images_input_folder, landmarks_shape=(33, 3)):
    pose_classes = sorted([n for n in os.listdir(images_input_folder) if not n.startswith('.')])
    pose_samples = []
    pose_embedder = FullBodyPoseEmbedder()

    for class_name in pose_classes:
        class_in_path = os.path.join(images_input_folder, class_name)
        if not os.path.isdir(class_in_path):
            continue

        image_names = sorted([n for n in os.listdir(class_in_path) if not n.startswith('.')])
        for image_name in tqdm.tqdm(image_names, desc=f'Processing {class_name}'):
            input_path = os.path.join(class_in_path, image_name)
            image = cv2.imread(input_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = image.shape[:2]

            with mp_solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose_tracker:
                result = pose_tracker.process(image=image_rgb)
                landmarks = result.pose_landmarks

            if landmarks is not None:
                flat_landmarks = flatten_landmark_features(landmarks, frame_width, frame_height)
                if len(flat_landmarks) == 99:
                    reshaped_landmarks = np.array(flat_landmarks, np.float32).reshape(landmarks_shape)

                    if class_name == "bad_form":
                        base_name = os.path.splitext(image_name)[0]

                        match = re.match(r"(.+?)(?:_\d+.*)?$", base_name)
                        sample_class_name = match.group(1) if match else base_name
                    else:
                        sample_class_name = class_name

                    pose_samples.append(PoseSample(
                        name=image_name,
                        class_name=sample_class_name,
                        landmarks=reshaped_landmarks,
                        embedding=pose_embedder(reshaped_landmarks),
                    ))
    return pose_samples

def bootstrap_from_folder(images_input_folder, images_output_folder, csvs_out_folder):
    pose_class_names = sorted([n for n in os.listdir(images_input_folder) if not n.startswith('.')])
    os.makedirs(csvs_out_folder, exist_ok=True)

    for class_name in pose_class_names:
        print(f'Bootstrapping: {class_name}')
        class_in_path = os.path.join(images_input_folder, class_name)
        class_out_path = os.path.join(images_output_folder, class_name)
        class_csv_path = os.path.join(csvs_out_folder, class_name + '.csv')
        os.makedirs(class_out_path, exist_ok=True)

        image_names = sorted([n for n in os.listdir(class_in_path) if not n.startswith('.')])

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
                        draw_landmarks_safe(
                            output_image,
                            landmarks,
                            mp_solutions.pose.POSE_CONNECTIONS,
                            landmark_color=(0, 255, 0),
                            connection_color=(0, 0, 255),
                            thickness=2,
                            radius=2
                        )
                output_bgr = rgb_to_bgr(output_image)
                output_image_path = os.path.join(class_out_path, image_name)
                cv2.imwrite(output_image_path, output_bgr)

                if result.pose_landmarks is not None:
                    flat_landmarks = flatten_landmark_features(
                    result.pose_landmarks,
                    frame_width,
                    frame_height)
                    if len(flat_landmarks) == 99:
                        writer.writerow([image_name] + list(map(str, flat_landmarks)))

                    # Also save the XZ projection
                    landmark_array = np.array(flat_landmarks, dtype=np.float32).reshape((33, 3))
                    xz_img = draw_xz_projection(output_bgr, landmark_array)
                    combined_img = np.concatenate((output_bgr, xz_img), axis=1)
                    cv2.imwrite(output_image_path, combined_img)

def align_images_and_csvs(images_output_folder, csvs_out_folder, print_removed_items=False):
    pose_classes = sorted([n for n in os.listdir(images_output_folder) if not n.startswith('.')])
    for class_name in pose_classes:
        class_out_folder = os.path.join(images_output_folder, class_name)
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

def flatten_landmark_features(landmarks, frame_width, frame_height):
    features = []
    for lm in landmarks.landmark:
        features.extend([lm.x * frame_width, lm.y * frame_height, lm.z * frame_width])
    return features
