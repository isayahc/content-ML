import os
from pytube import YouTube
from pytube.exceptions import RegexMatchError
from pytube.streams import Stream

def download_youtube_video(url: str, desired_quality: str = '360p', video_output: str = 'video.mp4') -> None:
    """
    Downloads a YouTube video in the specified quality.

    Args:
        url (str): The URL of the YouTube video.
        desired_quality (str): The desired quality of the video. Defaults to '360p'.
            Possible values: '144p', '240p', '360p', '480p', '720p', '1080p', '1440p', '2160p'.
        video_output (str): The name and location of the video file to be saved. Defaults to 'video.mp4'.

    Returns:
        None

    Raises:
        RegexMatchError: If the YouTube URL is invalid or the video is unavailable.
    """
    try:
        # Get YouTube video
        yt = YouTube(url)

        # Get streams of the desired quality
        streams = yt.streams.filter(progressive=True, file_extension='mp4', res=desired_quality)

        # Get the first stream (assuming it exists)
        stream = streams.first()

        # Check if a stream of the desired quality exists
        if stream is not None:
            # Get the directory and file name from the video_output
            output_directory, output_file = os.path.split(video_output)

            # Create the directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)

            # Download the video
            stream.download(output_path=output_directory, filename=output_file)
            print("Video downloaded successfully!")
        else:
            print(f"No video found with the desired quality: {desired_quality}")

    except RegexMatchError:
        print("Invalid YouTube URL or video unavailable.")
