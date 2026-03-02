#!/usr/bin/env python3
"""
Audio Transcriber & Summarizer - Main Entry Point
"""

import os
import sys
import json
import argparse
import threading
import time
import gc
import subprocess
import multiprocessing as mp
import faulthandler
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Workaround for OpenMP duplicate library on macOS/conda
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TQDM_DISABLE", "1")

# Use 'spawn' for multiprocessing to avoid crashes on macOS
try:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
except RuntimeError:
    pass

faulthandler.enable()

# Base logic directory
BASE_DIR = Path(__file__).parent.parent.parent.absolute()

# Local imports
try:
    from skills.transcription import audio_utils
    from skills.transcription import analysis
except ImportError:
    # Fallback for different run contexts
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from skills.transcription import audio_utils
    from skills.transcription import analysis

def _stop_tqdm_monitor():
    try:
        import tqdm
        inst = getattr(tqdm.tqdm, '_instances', None)
        if inst is not None:
            inst.clear()
    except:
        pass

class AudioTranscriberSummarizer:
    def __init__(self, model_size="base", language="auto"):
        from faster_whisper import WhisperModel
        print(f"🤖 Inicializando Audio Transcriber & Summarizer (faster-whisper)...")
        print(f"📡 Cargando modelo '{model_size}' ...")

        preferred_compute_types = ["float16", "int8_float16", "int8"]
        model_loaded = False
        last_error = None
        for compute_type in preferred_compute_types:
            try:
                self.model = WhisperModel(model_size, device="auto", compute_type=compute_type)
                print(f"✅ Modelo cargado con compute_type={compute_type}")
                model_loaded = True
                break
            except Exception as e:
                last_error = e
                continue
        if not model_loaded:
            raise RuntimeError(f"No se pudo cargar el modelo {model_size}: {last_error}")

        self.language = language
        self.transcribe_timeout = 300
        self.hf_token = os.environ.get("HF_TOKEN")
        analysis.setup_nltk()
        print("✅ Inicialización completada")
    
    def transcribe_audio(self, audio_path: str) -> Optional[Dict]:
        print(f"🎤 Transcribiendo audio: {os.path.basename(audio_path)}")
        t_start = time.time()
        
        try:
            transcribe_kwargs = {"vad_filter": True, "beam_size": 5}
            if self.language != "auto":
                transcribe_kwargs["language"] = self.language

            call_result = {}
            def _call_transcribe():
                try:
                    call_result['value'] = self.model.transcribe(audio_path, **transcribe_kwargs)
                except Exception as e:
                    call_result['exc'] = e

            trans_thread = threading.Thread(target=_call_transcribe, daemon=True)
            trans_thread.start()
            trans_thread.join(timeout=self.transcribe_timeout)

            if trans_thread.is_alive() or 'exc' in call_result:
                print(f"❌ Error o timeout en transcripción.")
                return None

            segments_iter, info = call_result.get('value')
            segments = []
            full_text_parts = []
            for i, seg in enumerate(segments_iter):
                seg_text = seg.text.strip()
                segments.append({
                    "id": i,
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg_text,
                })
                full_text_parts.append(seg_text)

            full_text = " ".join(full_text_parts).strip()
            return {
                "text": full_text,
                "language": getattr(info, 'language', "unknown"),
                "duration": float(getattr(info, "duration", 0.0)),
                "segments": segments,
            }
        except Exception as e:
            print(f"❌ Error al transcribir: {e}")
            return None
        finally:
            gc.collect()

    def generate_social_descriptions(self, text: str) -> Dict[str, str]:
        """Wrapper for social media description generation to maintain compatibility."""
        return analysis.generate_social_descriptions(text, hf_token=self.hf_token)

    def process_media_file(self, file_path: str, output_dir: Optional[str] = None, 
                           keep_audio: bool = False, summary_sentences: int = 5) -> Optional[Dict]:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"❌ Error: El archivo {file_path} no existe")
            return None
        
        # Use relative paths by default
        if output_dir is None:
            output_dir = BASE_DIR / "transcripciones"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        file_extension = file_path.suffix.lower()
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma'}

        temp_audio_path = None
        if file_extension in video_extensions:
            temp_audio_path = output_dir / f"{file_path.stem}_temp_audio.wav"
            audio_path = audio_utils.extract_audio_from_video(str(file_path), str(temp_audio_path))
        elif file_extension in audio_extensions:
            if file_extension == '.wav':
                audio_path = str(file_path)
            else:
                temp_audio_path = output_dir / f"{file_path.stem}_temp_audio.wav"
                audio_path = audio_utils.extract_audio_from_audio(str(file_path), str(temp_audio_path))
        else:
            print(f"❌ Error: Formato no soportado: {file_extension}")
            return None
        
        if not audio_path: return None
        
        transcription = self.transcribe_audio(audio_path)
        if not transcription: return None
        
        clean_text = analysis.clean_text(transcription["text"])
        keywords = analysis.extract_keywords(clean_text)
        extractive_summary = analysis.create_extractive_summary(clean_text, summary_sentences)
        topic_summary = analysis.create_topic_summary(clean_text)
        social_descriptions = analysis.generate_social_descriptions(clean_text)
        
        result = {
            "file_info": {
                "original_file": str(file_path),
                "file_type": "video" if file_extension in video_extensions else "audio",
                "processed_date": datetime.now().isoformat(),
            },
            "transcription": {
                "language": transcription.get("language", "unknown"),
                "duration_seconds": transcription.get("duration", 0),
                "full_text": clean_text,
                "segments": transcription.get("segments", [])
            },
            "analysis": {
                "keywords": keywords,
                "extractive_summary": extractive_summary,
                "topic_summary": topic_summary,
                "social_descriptions": social_descriptions,
                "text_stats": {
                    "character_count": len(clean_text),
                    "word_count": len(clean_text.split()),
                }
            }
        }
        
        # Save results
        json_file = output_dir / f"{file_path.stem}_analysis.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        if social_descriptions:
            social_file = output_dir / f"{file_path.stem}_social.txt"
            with open(social_file, 'w', encoding='utf-8') as f:
                f.write("OPCIONES PARA REELS / SHORTS\n")
                f.write(f"Opción 1: {social_descriptions.get('filosofia', '')}\n")
                f.write(f"Opción 2: {social_descriptions.get('leccion', '')}\n")
                f.write(f"Opción 3: {social_descriptions.get('aprendizaje', '')}\n")

        # Cleanup
        if temp_audio_path and not keep_audio and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        elif temp_audio_path and keep_audio:
            final_audio_path = output_dir / f"{file_path.stem}_audio.wav"
            if temp_audio_path != final_audio_path:
                os.rename(temp_audio_path, final_audio_path)
        
        return result

def select_video_file():
    videos_dir = BASE_DIR / "videos"
    if not videos_dir.exists():
        print(f"❌ La carpeta {videos_dir} no existe")
        return None
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    video_files = [f for f in os.listdir(videos_dir) if any(f.lower().endswith(ext) for ext in video_extensions)]
    if not video_files: return None
    
    video_files.sort()
    print(f"\n🎬 ARCHIVOS DE VIDEO DISPONIBLES")
    for i, file in enumerate(video_files, 1):
        print(f"{i:2d}. {file}")
    
    try:
        choice = input(f"🎯 Selecciona un archivo (1-{len(video_files)}) o 0 para cancelar: ").strip()
        if choice == '0': return None
        idx = int(choice) - 1
        return str(videos_dir / video_files[idx])
    except:
        return None

def run_isolated(file_path: str, args):
    script_path = os.path.abspath(__file__)
    cmd = [sys.executable, script_path, file_path, '--child-run', '--model', args.model]
    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    subprocess.run(cmd, env=env)

def main():
    parser = argparse.ArgumentParser(description='Audio Transcriber & Summarizer')
    parser.add_argument('file_path', nargs='?', help='Ruta al archivo')
    parser.add_argument('--model', default='base', help='Whisper model')
    parser.add_argument('--language', default='auto', help='Language')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--child-run', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child_run:
        processor = AudioTranscriberSummarizer(args.model, args.language)
        processor.process_media_file(args.file_path)
        sys.exit(0)

    if args.interactive or not args.file_path:
        while True:
            file_path = select_video_file()
            if not file_path: break
            run_isolated(file_path, args)
            if input("\n¿Procesar otro? [Y/n]: ").lower().startswith('n'): break
    else:
        run_isolated(args.file_path, args)

if __name__ == "__main__":
    main()
