# YouTube Download Skill

**Description**: Robust YouTube video and audio downloader using `yt-dlp`.

## Capabilities

- **Video Download**: High-quality video downloads.
- **Audio Extraction**: Extract audio from video.
- **Authentication**: Usage of `cookies.txt` for age-restricted content.

## Dependencies
- `yt-dlp`
- `ffmpeg`

## Usage

```python
from skills.youtube_download.tool import YoutubeVideoDownloader

downloader = YoutubeVideoDownloader()
downloader.download("https://youtube.com/watch?v=...", output_path=".")
```
