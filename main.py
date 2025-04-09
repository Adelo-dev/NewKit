from inferencers.image_inferencer import ImageInference 
from inferencers.video_inferencer import VideoInferencer

def main():
    # image_inference = ImageInference(debug_mode=True)
    video_inference = VideoInferencer(debug_mode=True)
    # image_inference.inference(image_path='sample_data/sample.jpg', output_path='output/standing_result.jpg')
    video_inference.inference(stream_path="input/pushups.mp4",output_path='output/sample_result.mp4',show=True, should_infer=True )



if __name__ == "__main__":
    main()
