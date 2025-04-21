import csv
import os


# ------------------------- #
#   LANDMARKS CSV UTILS     #
# ------------------------- #

def generate_landmark_header():
    """
    Returns a CSV header for 33 pose landmarks:
    video_name, frame_number, landmark_0_x, ..., landmark_32_visibility
    """
    header = ["file_name", "frame_number"]
    for i in range(33):
        header += [
            f"landmark_{i}_x",
            f"landmark_{i}_y",
            f"landmark_{i}_z",
            f"landmark_{i}_visibility"
        ]
    return header


def write_pose_csv_row(csv_path, video_name, frame_number, landmark_data, write_header=False):
    """
    Writes a single row to a CSV containing MediaPipe pose landmark data.

    Args:
        csv_path (str): Path to the CSV file.
        video_name (str): Name of the video.
        frame_number (int): The current frame number.
        landmark_data (NormalizedLandmarkList): MediaPipe pose_landmarks object.
        write_header (bool): Whether to write the header (only on first call).
    """
    # Create directory if needed
    output_dir = os.path.dirname(csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow(generate_landmark_header())

        row = [video_name, frame_number]
        for lm in landmark_data.landmark:
            row.extend([
                round(lm.x, 5),
                round(lm.y, 5),
                round(lm.z, 5),
                round(lm.visibility, 5)
            ])

        writer.writerow(row)


# --------------------------- #
#   EMBEDDING CSV UTILS       #
# --------------------------- #

def generate_embedding_header(embedding_names):
    """
    Create CSV header using landmark pair names for 3D vector embedding.
    Each pair adds 3 columns: dx, dy, dz.
    """
    header = ["file_name", "frame_number"]
    for from_lmk, to_lmk in embedding_names:
        header.extend([
            f"{from_lmk}_to_{to_lmk}_dx",
            f"{from_lmk}_to_{to_lmk}_dy",
            f"{from_lmk}_to_{to_lmk}_dz"
        ])
    return header


def write_pose_embedding_csv_row(csv_path, video_name, frame_number, embedding, embedding_names=None, write_header=False):
    """
    Writes a row of pose embedding data (from FullBodyPoseEmbedder) to a CSV file.

    Args:
        csv_path (str): Output file path.
        video_name (str): Name of the source video or stream.
        frame_number (int): Frame index.
        embedding (np.ndarray): Output from FullBodyPoseEmbedder. Shape (N, 3).
        embedding_names (List[Tuple[str, str]]): Optional landmark names for header generation.
        write_header (bool): Whether to write the header row.
    """
    output_dir = os.path.dirname(csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if write_header and embedding_names:
            header = generate_embedding_header(embedding_names)
            writer.writerow(header)

        flat = embedding.flatten().tolist()
        writer.writerow([video_name, frame_number] + flat)


def write_joint_angles_csv_row(csv_path, video_name, frame_number, angle_dict, write_header=False):
    """
    Saves joint angle dictionary to CSV.
    Each joint angle is one column.
    """
    output_dir = os.path.dirname(csv_path)


    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if write_header:
            headers = ["file_name", "frame_number"] + list(angle_dict.keys())
            writer.writerow(headers)

        row = [video_name, frame_number] + list(angle_dict.values())
        writer.writerow(row)

