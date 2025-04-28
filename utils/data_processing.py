import os
import uuid
import zipfile
from pathlib import Path

import gdown


def download_dataset(url, save_path):
    """Download a dataset from a given URL and save it to the specified path and extract it."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_path = Path(save_path, str(uuid.uuid4()))
    gdown.download(url, str(output_path.with_suffix('.zip')), fuzzy=True, quiet=False)

    # Extract the dataset
    with zipfile.ZipFile(str(output_path.with_suffix('.zip')), 'r') as zip_ref:
        zip_ref.extractall(str(output_path))

    # Remove the zip file after extraction
    os.remove(output_path.with_suffix('.zip'))
    return str(output_path)
