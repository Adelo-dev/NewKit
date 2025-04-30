from data_processing.pose_classifier import PoseClassifier
from inferencers.image_inferencer import ImageInference
from inferencers.video_inferencer import VideoInferencer


def main():
    # image_inference = ImageInference(debug_mode=True)
    # image_inference.inference(
    #     image_path="data/sample_data/sample.jpg",
    #     output_path="data/output/standing_result.jpg"
    # )

    pose_classifier = PoseClassifier()
    pose_samples_file = pose_classifier.generate_pose_samples(images_input_folder="data/sample_data/classify_sample",
                                          output_folder="data/datasets")
    video_inference = VideoInferencer(debug_mode=True)
    video_inference.inference(
        stream_path="data/sample_data/sample_knee_pushups.mp4",
        output_path="data/output",
        show=False,
        should_infer=True,
        classifier_inputs=pose_samples_file
    )

if __name__ == "__main__":
    main()
