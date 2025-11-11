from typing import Optional
from yt_dlp import YoutubeDL
from pydub import AudioSegment
from typing import Optional
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import os

class YouTubeDownloader:
    def __init__(self, ffmpeg_path: Optional[str] = None):
        """
        Initialize the YouTube Downloader.
        
        Args:
            ffmpeg_path: Optional path to ffmpeg executable
        """
        if ffmpeg_path:
            os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
            AudioSegment.ffmpeg = ffmpeg_path
            AudioSegment.ffprobe = ffmpeg_path.replace('ffmpeg', 'ffprobe')
        
        self.ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': '%(title)s.%(ext)s',
            'merge_output_format': 'mp4',
            'cookiesfrombrowser': ('chrome',),
            'quiet': False,
        }

    def download_video(self, url: str) -> None:
        """
        Download a video from YouTube.
        
        Args:
            url: YouTube video URL
        """
        try:
            with YoutubeDL(self.ydl_opts) as ydl:
                print(f"Downloading: {url}")
                ydl.download([url])
                print("Download completed.")
        except Exception as e:
            print(f"Error downloading video: {e}")
            raise

    def convert_to_mp3(self, input_file: str, output_file: Optional[str] = None, 
                      bitrate: str = "192k") -> None:
        """
        Convert a video file to MP3.
        
        Args:
            input_file: Path to input video file
            output_file: Optional output MP3 filename
            bitrate: Audio bitrate (default: "192k")
        """
        if not output_file:
            output_file = os.path.splitext(input_file)[0] + '.mp3'
            
        try:
            audio = AudioSegment.from_file(input_file)
            audio.export(output_file, format="mp3", bitrate=bitrate)
            print(f"Audio converted successfully to {output_file}")
        except Exception as e:
            print(f"Error during conversion: {e}")
            raise

    def process_url(self, url: str, convert_to_audio: bool = False, 
                   delete_video: bool = False) -> None:
        """
        Download a video and optionally convert to MP3.
        
        Args:
            url: YouTube video URL
            convert_to_audio: Whether to convert to MP3
            delete_video: Whether to delete the video file after conversion
        """
        try:
            # Download video
            self.download_video(url)
            
            if convert_to_audio:
                # Get the downloaded video filename
                video_title = None
                with YoutubeDL(self.ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_title = ydl.prepare_filename(info)
                
                # Convert to MP3
                if video_title:
                    self.convert_to_mp3(video_title)
                    
                    # Delete video if requested
                    if delete_video and os.path.exists(video_title):
                        os.remove(video_title)
                        print(f"Deleted video file: {video_title}")
                        
        except Exception as e:
            print(f"Error processing URL: {e}")
            raise
