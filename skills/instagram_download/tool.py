"""
instagram_downloader.py
Descarga videos de Instagram dado un enlace.
Requiere: pip install instaloader
"""

import instaloader
import sys
import os
import re

def download_instagram_video(url, output_dir="videos"):
    """
    Descarga un video de Instagram dado el enlace.
    Args:
        url (str): Enlace al post de Instagram
        output_dir (str): Carpeta de destino
    Returns:
        str: Ruta del archivo descargado o None si falla
    """
    # Validar el enlace
    pattern = r"https?://(www\.)?instagram.com/(p|reel|tv)/[\w-]+"
    if not re.match(pattern, url):
        print("❌ Enlace de Instagram no válido.")
        return None

    # Crear carpeta de salida si no existe
    # Usar carpeta 'videos' por defecto si no existe
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Inicializar Instaloader
    loader = instaloader.Instaloader(dirname_pattern=output_dir,
                                     save_metadata=False,
                                     download_video_thumbnails=False,
                                     download_geotags=False,
                                     post_metadata_txt_pattern="")
    try:
        print(f"⬇️ Descargando video de: {url}")
        post = instaloader.Post.from_shortcode(loader.context, url.split('/')[-2])
        loader.download_post(post, target=output_dir)
        print("✅ Descarga completada.")
        return output_dir
    except Exception as e:
        print(f"❌ Error al descargar: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        url = input("Introduce el enlace de Instagram: ").strip()
        if not url:
            print("❌ No se proporcionó ningún enlace.")
            sys.exit(1)
        output_dir = "videos"
    else:
        url = sys.argv[1]
        output_dir = "videos"
    download_instagram_video(url, output_dir)
