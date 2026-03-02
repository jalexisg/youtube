import os
from pathlib import Path
from typing import Optional

def extract_audio_from_video(video_path: str, audio_path: Optional[str] = None) -> Optional[str]:
    """
    Extrae audio de un archivo de video usando MoviePy.
    """
    try:
        from moviepy import VideoFileClip
    except ImportError:
        print("❌ Error: moviepy no está instalado.")
        return None

    if audio_path is None:
        video_name = Path(video_path).stem
        audio_path = str(Path(video_path).parent / f"{video_name}_audio.wav")
    
    print(f"🎬 Extrayendo audio de: {os.path.basename(video_path)}")
    
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path, logger=None)
        audio.close()
        video.close()
        
        print(f"🎵 Audio extraído: {os.path.basename(audio_path)}")
        return audio_path
        
    except Exception as e:
        print(f"❌ Error al extraer audio: {e}")
        return None

def extract_audio_from_audio(input_audio_path: str, output_audio_path: Optional[str] = None) -> Optional[str]:
    """
    Procesa un archivo de audio usando PyDub (conviértelo a wav).
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        print("❌ Error: pydub no está instalado.")
        return None

    if output_audio_path is None:
        audio_name = Path(input_audio_path).stem
        output_audio_path = str(Path(input_audio_path).parent / f"{audio_name}_processed.wav")
    
    print(f"🎵 Procesando audio: {os.path.basename(input_audio_path)}")
    
    try:
        audio = AudioSegment.from_file(input_audio_path)
        audio.export(output_audio_path, format="wav")
        
        print(f"✅ Audio procesado: {os.path.basename(output_audio_path)}")
        return output_audio_path
        
    except Exception as e:
        print(f"❌ Error al procesar audio: {e}")
        return None
