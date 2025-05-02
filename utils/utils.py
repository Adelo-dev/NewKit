import csv
import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def write_pose_embedding_csv_row(csv_path: str, embedding):
    output_dir = os.path.dirname(csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    embedding_flat_list = embedding.flatten().tolist()
    df = pd.DataFrame([embedding_flat_list])

    # Check if the file exists to determine if we need to write the header
    file_exists = os.path.exists(csv_path)

    if not file_exists:
        df.to_csv(csv_path, mode='w', header=True, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def dump_for_the_app():
    pose_samples_folder = 'fitness_poses_csvs_out'
    pose_samples_csv_path = 'combined_fitness_poses_csv_out/fitness_poses_csvs_out.csv'
    file_extension = 'csv'
    file_separator = ','

    # Each file in the folder represents one pose class.
    file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

    with open(pose_samples_csv_path, 'w') as csv_out:
        csv_out_writer = csv.writer(csv_out, delimiter=file_separator, quoting=csv.QUOTE_MINIMAL)
        for file_name in file_names:
        # Use file name as pose class name.
            class_name = file_name[:-(len(file_extension) + 1)]

        # One file line: `sample_00001,x1,y1,x2,y2,....`.
            with open(os.path.join(pose_samples_folder, file_name)) as csv_in:
                csv_in_reader = csv.reader(csv_in, delimiter=file_separator)
                for row in csv_in_reader:
                    row.insert(1, class_name)
                    csv_out_writer.writerow(row)

def get_class_name(file_name: str, file_extension: str) -> str:
    if file_name.endswith(f".{file_extension}"):
        return file_name[:-(len(file_extension) + 1)]
    else:
        raise ValueError(f"File '{file_name}' does not end with .{file_extension}")

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def video_to_frames(video_path: str, output_folder: str, frame_rate: int = 30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Save the frame as an image file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    video_capture.release()

def create_confusion_matrix(data, output_path=None):
    y_true, y_pred = zip(*data, strict=False)

    # Get all unique labels to ensure consistent ordering
    labels = sorted(set(y_true + y_pred))

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

