"""
extract_audio_from_video.py
Extrae el audio de un video en la carpeta 'videos'.
Requiere: pip install pydub
"""

import os
from pydub import AudioSegment

def list_videos(directory="videos"):
    return [f for f in os.listdir(directory) if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov'))]

def extract_audio():
    video_dir = "videos"
    if not os.path.exists(video_dir):
        print(f"La carpeta '{video_dir}' no existe.")
        return
    videos = list_videos(video_dir)
    if not videos:
        print(f"No hay videos en la carpeta '{video_dir}'.")
        return
    print("Videos disponibles:")
    for idx, v in enumerate(videos, 1):
        print(f"{idx}. {v}")
    try:
        choice = int(input("Selecciona el número del video para extraer el audio: "))
        if not (1 <= choice <= len(videos)):
            print("Selección inválida.")
            return
        video_file = os.path.join(video_dir, videos[choice - 1])
        audio_dir = "audios"
        os.makedirs(audio_dir, exist_ok=True)
        audio_file = os.path.join(audio_dir, os.path.splitext(videos[choice - 1])[0] + ".mp3")
        print(f"Extrayendo audio de '{video_file}'...")
        AudioSegment.ffmpeg = "/usr/local/bin/ffmpeg"
        video = AudioSegment.from_file(video_file)
        video.export(audio_file, format="mp3")
        print(f"✅ Audio guardado como '{audio_file}'")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    extract_audio()
