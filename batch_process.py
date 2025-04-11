import os
from glob import glob

from tqdm import tqdm

from inferencers.image_inferencer import ImageInference
from inferencers.video_inferencer import VideoInferencer


def main():
    image_inference = ImageInference()
    video_inference = VideoInferencer()
    for image in tqdm(glob("input/*.jpg"), desc="Processing images"):
        image_inference.inference(image_path=image,
                                  output_path=f"output/{os.path.basename(image)}",
                                  show=False)

    for video in tqdm(glob("input/*.mp4"), desc="Processing videos"):
        video_inference.inference(stream_path=video,
                                  output_path=f"output/{os.path.basename(video)}",
                                  show=False)

if __name__ == "__main__":
    main()
