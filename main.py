# from inferencers.image_inferencer import ImageInference
from inferencers.video_inferencer import VideoInferencer


def main():
    # image_inference = ImageInference(debug_mode=True)
    # image_inference.inference(
    #     image_path="data/sample_data/sample.jpg",
    #     output_path="data/output/standing_result.jpg"
    # )

    video_inference = VideoInferencer(debug_mode=True)
    video_inference.inference(
        stream_path="trainee_videos/legraises_7.mp4",
        trainer_videos="trainer_videos",
        output_path="data/output",
        show=True,
        should_infer=True,
        classifier_errors="data/exercises/legraises/legraises_errors.csv",
        classifier_rep_count="data/exercises/legraises/legraises_rep_count.csv",
        add_new_data=True,
        exercise_name="legraises"
    )

if __name__ == "__main__":
    main()
