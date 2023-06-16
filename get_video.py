from pytube import YouTube
YouTube('https://www.youtube.com/watch?v=dQw4w9WgXcQ').streams.get_highest_resolution().download(filename='video.mp4')