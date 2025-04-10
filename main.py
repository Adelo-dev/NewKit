from inferencers.image_inferencer import ImageInference
from inferencers.video_inferencer import VideoInferencer


def main():
    image_inference = ImageInference(debug_mode=True)
    video_inference = VideoInferencer(debug_mode=True)
    image_inference.inference(
        image_path="sample_data/sample.jpg", output_path="output/standing_result.jpg"
    )
    video_inference.inference(
        stream_path="sample_data/sample_dips.mp4",
        output_path="output/sample_result.mp4", should_infer=True
    )


if __name__ == "__main__":
    main()
