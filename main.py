from inferencers.image_inferencer import ImageInference 
from inferencers.video_inferencer import VideoInferencer

def main():
    image_inference = ImageInference(debug_mode=True)
    #video_inference = VideoInferencer()
    image_inference.inference(image_path='sample_data/sample.jpg', output_path='output/standing_result.jpg')
    #video_inference.inference('input/pushups.mp4', 'output/pushups_result.mp4', should_infer=True)


if __name__ == "__main__":
    main()
