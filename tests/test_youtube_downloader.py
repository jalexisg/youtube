import unittest
from unittest.mock import patch, MagicMock
import os
from src.youtube_downloader import YouTubeDownloader

class TestYouTubeDownloader(unittest.TestCase):
    def setUp(self):
        self.downloader = YouTubeDownloader()
        self.test_url = "https://www.youtube.com/watch?v=test"
        
    @patch('yt_dlp.YoutubeDL')
    def test_download_video(self, mock_ydl):
        # Set up mock
        mock_ydl_instance = MagicMock()
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance
        
        # Test successful download
        self.downloader.download_video(self.test_url)
        mock_ydl_instance.download.assert_called_once_with([self.test_url])
        
        # Test failed download
        mock_ydl_instance.download.side_effect = Exception("Download failed")
        with self.assertRaises(Exception):
            self.downloader.download_video(self.test_url)
    
    @patch('pydub.AudioSegment')
    def test_convert_to_mp3(self, mock_audio_segment):
        # Set up mock
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        
        # Test successful conversion
        input_file = "test_video.mp4"
        output_file = "test_audio.mp3"
        self.downloader.convert_to_mp3(input_file, output_file)
        mock_audio_segment.from_file.assert_called_once_with(input_file)
        mock_audio.export.assert_called_once_with(output_file, format="mp3", bitrate="192k")
        
        # Test failed conversion
        mock_audio_segment.from_file.side_effect = Exception("Conversion failed")
        with self.assertRaises(Exception):
            self.downloader.convert_to_mp3(input_file, output_file)

if __name__ == '__main__':
    unittest.main()
