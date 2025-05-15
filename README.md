# YouTube Media Downloader

A Python-based tool for downloading and converting YouTube videos with support for various media formats. This project uses yt-dlp for downloading and pydub for audio processing.

## 🚨 Legal Notice

This tool is for educational purposes only. Users are responsible for complying with YouTube's Terms of Service and respecting copyright laws. Only download content that you have permission to use.

## ✨ Features

- Download YouTube videos in highest quality
- Convert video files to audio (MP3)
- Support for various video formats
- Custom output filename templates
- Browser cookie integration
- Progress tracking
- Error handling

## 🛠️ Requirements

- Python 3.8+
- FFmpeg
- Required Python packages:
  ```
  yt-dlp
  pydub
  moviepy
  ```

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/jalexisg/youtube.git
cd youtube
```

2. Install FFmpeg:
   - On macOS:
     ```bash
     brew install ffmpeg
     ```
   - On Ubuntu/Debian:
     ```bash
     sudo apt-get install ffmpeg
     ```
   - On Windows:
     - Download from [FFmpeg official website](https://ffmpeg.org/download.html)
     - Add to system PATH

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Using the Jupyter Notebook

1. Launch Jupyter:
```bash
jupyter notebook
```

2. Open `youtubeDownloader.ipynb`
3. Follow the interactive cells

### Video Download Function

```python
from yt_dlp import YoutubeDL

def download_youtube_video(url):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',
        'merge_output_format': 'mp4',
        'cookiesfrombrowser': ('chrome',),
        'quiet': False,
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
```

### Audio Conversion

```python
from pydub import AudioSegment

# Convert MKV to MP3
audio = AudioSegment.from_file("video.mkv")
audio.export("audio.mp3", format="mp3", bitrate="192k")
```

## 🔧 Configuration

- FFmpeg path settings in environment
- Custom output templates
- Browser cookie integration
- Audio quality settings
- Video format preferences

## 📝 Notes

- Ensure proper FFmpeg installation
- Check Internet connection stability
- Verify storage space availability
- Monitor download progress
- Handle large files appropriately

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues:

1. FFmpeg not found:
   - Verify FFmpeg installation
   - Check system PATH
   - Set explicit paths in code

2. Download errors:
   - Check internet connection
   - Verify video availability
   - Update yt-dlp

3. Conversion issues:
   - Verify file permissions
   - Check disk space
   - Update FFmpeg

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
