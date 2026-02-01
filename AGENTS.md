# AGENTS.md

> [!NOTE]
> This file is a manifesto for AI agents working on this project. It defines the project's purpose, structure, and available "skills" to ensure consistent and intelligent assistance.

## Project Overview

**Name**: Audio Transcriber & Summarizer (Web & CLI)
**Description**: An advanced Python project to extract audio from videos (YouTube, Instagram), transcribe content using OpenAI's Whisper, and generate intelligent summaries. It features a FastAPI web interface and Docker support.

## 🛠️ Agent Skills

This project is organized into "Skills" located in the `skills/` directory. Each skill is a self-contained module with its own `SKILL.md` instruction file.

### [Transcription](skills/transcription/SKILL.md)
**Path**: `skills/transcription`
**Description**: Core capability to transcribe audio files using Whisper and generate summaries.
**Key Files**: `tool.py` (formerly `audio_transcriber_summarizer.py`)

### [YouTube Downloader](skills/youtube_download/SKILL.md)
**Path**: `skills/youtube_download`
**Description**: Downloads videos and audio from YouTube using `yt-dlp`. Handles cookies and age restrictions.
**Key Files**: `tool.py` (formerly `youtube_video_downloader.py`)

### [Instagram Downloader](skills/instagram_download/SKILL.md)
**Path**: `skills/instagram_download`
**Description**: Downloads content from Instagram.
**Key Files**: `tool.py` (formerly `instagram_downloader.py`)

### [Utils](skills/utils/SKILL.md)
**Path**: `skills/utils`
**Description**: General utility functions for media processing.
**Key Files**: `extract_audio.py`, `image_resizer.py`

## 💻 Development Guidelines

### Environment
- **Docker**: The preferred way to run the app is via `docker-compose up`.
- **Local**: Requires Python 3.8+ and FFmpeg.

### Project Structure
```
/
├── AGENTS.md           # This file
├── web_server.py       # FastAPI Entrypoint
├── skills/             # Agent Skills Modules
│   ├── transcription/
│   ├── youtube_download/
│   ├── instagram_download/
│   └── utils/
├── videos/             # Downloaded media
├── transcripciones/    # Output files
└── web/                # Frontend assets
```

### Conventions
- **Imports**: modifying imports requires checking references in `web_server.py`.
- **Artifacts**: Agents should log major architectural changes in `AGENTS.md`.
