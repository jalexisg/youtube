# Graph Report - youtube  (2026-04-30)

## Corpus Check
- 21 files · ~52,521 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 97 nodes · 100 edges · 10 communities detected
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]

## God Nodes (most connected - your core abstractions)
1. `AudioTranscriberSummarizer` - 10 edges
2. `YouTubeDownloader` - 7 edges
3. `main()` - 5 edges
4. `get_transcriber()` - 4 edges
5. `TestSocialDescriptions` - 4 edges
6. `create_topic_summary()` - 4 edges
7. `URLRequest` - 3 edges
8. `setLoading()` - 3 edges
9. `showResult()` - 3 edges
10. `TestYouTubeDownloader` - 3 edges

## Surprising Connections (you probably didn't know these)
- `get_transcriber()` --calls--> `AudioTranscriberSummarizer`  [INFERRED]
  web_server.py → skills/transcription/tool.py
- `URLRequest` --uses--> `AudioTranscriberSummarizer`  [INFERRED]
  web_server.py → skills/transcription/tool.py
- `TestSocialDescriptions` --uses--> `AudioTranscriberSummarizer`  [INFERRED]
  tests/test_social_descriptions.py → skills/transcription/tool.py
- `TestYouTubeDownloader` --uses--> `YouTubeDownloader`  [INFERRED]
  tests/test_youtube_downloader.py → src/youtube_downloader.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.18
Nodes (6): TestSocialDescriptions, AudioTranscriberSummarizer, main(), Wrapper for social media description generation to maintain compatibility., run_isolated(), select_video_file()

### Community 1 - "Community 1"
Cohesion: 0.18
Nodes (12): clean_text(), create_extractive_summary(), create_topic_summary(), extract_keywords(), generate_social_descriptions(), Genera 3 opciones de descripción para Reels/Shorts usando Hugging Face., Configura las dependencias de NLTK, Limpia y normaliza el texto transcrito (+4 more)

### Community 2 - "Community 2"
Cohesion: 0.19
Nodes (6): Initialize the YouTube Downloader.                  Args:             ffmpeg_pat, Download a video from YouTube.                  Args:             url: YouTube v, Convert a video file to MP3.                  Args:             input_file: Path, Download a video and optionally convert to MP3.                  Args:, YouTubeDownloader, TestYouTubeDownloader

### Community 3 - "Community 3"
Cohesion: 0.29
Nodes (6): capitalize(), formatText(), setLoading(), showError(), showResult(), updateStatus()

### Community 4 - "Community 4"
Cohesion: 0.28
Nodes (5): BaseModel, get_transcriber(), process_file_task(), process_url_task(), URLRequest

### Community 5 - "Community 5"
Cohesion: 0.4
Nodes (4): extract_audio_from_audio(), extract_audio_from_video(), Procesa un archivo de audio usando PyDub (conviértelo a wav)., Extrae audio de un archivo de video usando MoviePy.

### Community 6 - "Community 6"
Cohesion: 0.5
Nodes (4): process_images_in_folder(), Process all images in the input folder and save them to the output folder, Resize an image to 16:9 aspect ratio while maintaining its quality, resize_to_16_9()

### Community 9 - "Community 9"
Cohesion: 0.67
Nodes (3): extract_audio(), list_videos(), extract_audio_from_video.py Extrae el audio de un video en la carpeta 'videos'.

### Community 10 - "Community 10"
Cohesion: 0.5
Nodes (3): download_instagram_video(), instagram_downloader.py Descarga videos de Instagram dado un enlace. Requiere: p, Descarga un video de Instagram dado el enlace.     Args:         url (str): Enla

### Community 11 - "Community 11"
Cohesion: 0.67
Nodes (1): youtube_video_downloader.py Descarga videos de YouTube solicitando el link y los

## Knowledge Gaps
- **19 isolated node(s):** `Configura las dependencias de NLTK`, `Limpia y normaliza el texto transcrito`, `Extrae palabras clave del texto`, `Crea un resumen extractivo basado en las oraciones más importantes`, `Crea un resumen organizado por temas principales` (+14 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 11`** (3 nodes): `tool.py`, `download_youtube_video()`, `youtube_video_downloader.py Descarga videos de YouTube solicitando el link y los`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AudioTranscriberSummarizer` connect `Community 0` to `Community 4`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `get_transcriber()` connect `Community 4` to `Community 0`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `AudioTranscriberSummarizer` (e.g. with `URLRequest` and `TestSocialDescriptions`) actually correct?**
  _`AudioTranscriberSummarizer` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `YouTubeDownloader` (e.g. with `TestYouTubeDownloader` and `.setUp()`) actually correct?**
  _`YouTubeDownloader` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Configura las dependencias de NLTK`, `Limpia y normaliza el texto transcrito`, `Extrae palabras clave del texto` to the rest of the system?**
  _19 weakly-connected nodes found - possible documentation gaps or missing edges._