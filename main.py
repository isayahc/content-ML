from get_video import download_youtube_video
from segment_people import save_roi_images
import torch

def main():
    video_path = "./sample_video_0.mp4"
    output_dir = "./assets/video1"

    # download_youtube_video(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ&pp=ygUJcmljayByb2xs", video_output=video_path,desired_quality="720p",)
    
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    save_roi_images(model=model,video_path=video_path,output_dir=output_dir)
    x=0

if __name__ == '__main__':
    main()