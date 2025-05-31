import os
from pathlib import Path

import cv2
import mediapipe.python.solutions as mp_solutions
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from data_processing.pose_embedding import FullBodyPoseEmbedder
from data_processing.utils import flatten_landmark_features


def get_best_indices(signal, labels, cluster_id, cluster_center, top_k):
    return sorted(
        [(i, abs(signal[i] - cluster_center)) for i in range(len(signal)) if labels[i] == cluster_id],
        key=lambda x: x[1]
    )[:top_k]


def save_frames(pose_data, frame_data, save_dir, prefix):
    for i, (idx, _) in enumerate(frame_data):
        frame = pose_data[idx]["frame"]
        video_name = pose_data[idx]["video_name"]
        frame_number = pose_data[idx]["frame_num"]
        filename = save_dir / f"{prefix}_{video_name}_{i+1:02d}_frame_{frame_number:06d}.jpg"
        cv2.imwrite(str(filename), frame)


def extract_best_up_down_frames_from_folder(videos_folder, output_dir="frames", exercise_name="exercise", top_k=40):
    mp_pose = mp_solutions.pose
    pose_embedder = FullBodyPoseEmbedder()

    # Define final save paths
    output_dir = Path(output_dir) / exercise_name / "good_form"
    up_dir = output_dir / f"{exercise_name}_up"
    down_dir = output_dir / f"{exercise_name}_down"
    up_dir.mkdir(parents=True, exist_ok=True)
    down_dir.mkdir(parents=True, exist_ok=True)

    video_files = [f for f in os.listdir(videos_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(videos_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            continue

        pose_data = []
        frame_index = 0
        with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose_tracker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose_tracker.process(rgb)

                if result.pose_landmarks:
                    flat_landmarks = flatten_landmark_features(result.pose_landmarks, width, height)
                    if len(flat_landmarks) == 99:
                        reshaped = np.array(flat_landmarks, dtype=np.float32).reshape((33, 3))
                        embedding = pose_embedder(reshaped)
                        center_y = np.mean(reshaped[:, 1])

                        pose_data.append({
                            "frame": frame,
                            "embedding": embedding,
                            "center_y": center_y,
                            "frame_num": frame_index,
                            "video_name": Path(video_file).stem
                        })

                frame_index += 1
        cap.release()

        if not pose_data:
            print(f"No valid pose data in: {video_file}")
            continue

        signal = [d["center_y"] for d in pose_data]
        signal_np = np.array(signal).reshape(-1, 1)

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(signal_np)
        centers = kmeans.cluster_centers_.flatten()

        up_cluster = 0 if centers[0] < centers[1] else 1
        down_cluster = 1 - up_cluster

        best_up = get_best_indices(signal, labels, up_cluster, centers[up_cluster], top_k)
        best_down = get_best_indices(signal, labels, down_cluster, centers[down_cluster], top_k)

        save_frames(pose_data, best_up, up_dir, prefix="up")
        save_frames(pose_data, best_down, down_dir, prefix="down")

        print(f"{video_file}: Saved {len(best_up)} up and {len(best_down)} down frames.")

    print(f"\n✅ All videos processed. Total output saved to:\n{up_dir}\n{down_dir}")
