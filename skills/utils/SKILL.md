# Utils Skill

**Description**: Common utilities for media manipulation.

## Capabilities

- **Image Resizing**: Resize (and crop) images to specific aspect ratios (e.g., 9:16 for Reels/Shorts).
- **Audio Extraction**: Extract audio track from video files using FFmpeg.

## Dependencies
- `Pillow` (for images)
- `ffmpeg-python` (or `subprocess` calling ffmpeg)

## Usage

```python
from skills.utils.extract_audio import extract_audio
from skills.utils.image_resizer import resize_image
```
