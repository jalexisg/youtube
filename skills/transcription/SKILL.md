# Transcription Skill

**Description**: Provides functionality to transcrible audio and video files using OpenAI's Whisper model (via `faster-whisper`), and generate summaries using OpenAI's GPT models or locally.

## Capabilities

- **Transcribe**: Converts audio to text with timestamps.
- **Summarize**: Generates bullet-point summaries of the transcription.
- **Diarization**: (Optional) Can identify different speakers.

## Dependencies
- `faster-whisper`
- `openai` (for summarization)
- `ffmpeg` (system dependency)

## Usage

```python
from skills.transcription.tool import AudioTranscriberSummarizer

ats = AudioTranscriberSummarizer(model_size="base")
result = ats.process_media_file("input.mp4", output_dir="results/")
```
