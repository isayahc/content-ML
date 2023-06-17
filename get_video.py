from pytube import YouTube

# specify the quality of the video
desired_quality = '360p'  # set to the quality you want (e.g., '144p', '240p', '360p', '480p', '720p', '1080p')

# get YouTube video
yt = YouTube('https://www.youtube.com/watch?v=dQw4w9WgXcQ')

# get streams of the desired quality
streams = yt.streams.filter(progressive=True, file_extension='mp4', res=desired_quality)

# get the first stream (assuming it exists)
stream = streams.first()

# check if a stream of the desired quality exists
if stream is not None:
    # download the video
    stream.download(filename='video.mp4')
else:
    print(f"No video found with the desired quality: {desired_quality}")
