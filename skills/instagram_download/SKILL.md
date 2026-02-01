# Instagram Download Skill

**Description**: Downloads content from Instagram.

## Capabilities

- **Post Download**: Downloads videos/images from posts.
- **Reel Download**: Downloads Reels.

## Dependencies
- `instaloader` (or similar, depending on implementation)

## Usage

```python
from skills.instagram_download.tool import InstagramDownloader

downloader = InstagramDownloader()
downloader.download("https://instagram.com/p/...", output_path=".")
```
