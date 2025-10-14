"""
youtube_video_downloader.py
Descarga videos de YouTube solicitando el link y los guarda en la carpeta 'videos'.
Requiere: pip install yt-dlp
"""

import os
from yt_dlp import YoutubeDL

def download_youtube_video():
    url = input("Introduce el enlace de YouTube: ")
    print(f"⬇️ Descargando: {url}")
    output_dir = "videos"
    os.makedirs(output_dir, exist_ok=True)
    cookies_path = os.path.join(os.path.dirname(__file__), "cookies.txt")
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'quiet': False,
        'noplaylist': True,
    }
    if os.path.isfile(cookies_path):
        ydl_opts['cookies'] = cookies_path
        print("🔑 Usando cookies.txt para autenticación.")
    else:
        print("⚠️ No se encontró cookies.txt. Si tienes problemas, exporta tus cookies de YouTube.")
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("✅ Descarga completada.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    download_youtube_video()

if __name__ == "__main__":
    download_youtube_video()
