#!/usr/bin/env python3
"""
Audio Transcriber & Summarizer - Extrae audio de videos, transcribe y resume el contenido
Autor: Assistant
Fecha: Junio 2025
"""

import os
# Evita el choque de runtimes OpenMP (libiomp5.dylib duplicada) entre MKL (NumPy de conda)
# y PyTorch/Whisper instalados vía pip. Es un workaround seguro para este caso.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Try to disable tqdm monitor threads started by some libraries which can
# interact poorly with native extensions and cause segfaults on shutdown.
# This is a best-effort hint; some libs may not honor it, but it's harmless.
os.environ.setdefault("TQDM_DISABLE", "1")
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import re
from typing import List, Dict, Optional
import threading
import time
import gc
import subprocess
import multiprocessing as mp
import faulthandler

# On macOS, forking a process after threads or native libs are loaded
# can cause crashes (segfaults) and leaked semaphores. Use 'spawn'
# start method when possible. Do this early, before any child
# processes/threads are created.
try:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
except RuntimeError:
    # start method already set by the runtime; ignore
    pass

# Enable faulthandler to get Python tracebacks on segfaults
faulthandler.enable()


def _stop_tqdm_monitor():
    """
    Attempt to clean up tqdm monitor thread by clearing tqdm instances.
    This is a best-effort no-raise helper to reduce segfaults caused by
    lingering tqdm monitor threads from native extensions.
    """
    try:
        import tqdm
        # _instances is a WeakSet; clear it to drop references to progress bars
        inst = getattr(tqdm.tqdm, '_instances', None)
        if inst is not None:
            try:
                inst.clear()
            except Exception:
                # ignore any errors; this is best-effort
                pass
    except Exception:
        pass

# Import necessary libraries
def check_and_import_dependencies():
    """Verifica e importa dependencias necesarias (versión faster-whisper)."""
    missing_deps = []

    # faster-whisper (sustituye a openai-whisper + torch)
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper importado correctamente")
    except ImportError:
        WhisperModel = None
        missing_deps.append("faster-whisper")

    # MoviePy
    try:
        from moviepy import VideoFileClip
        print("✅ MoviePy importado correctamente")
    except ImportError as e:
        VideoFileClip = None
        print(f"❌ Error con MoviePy: {e}")
        missing_deps.append("moviepy")

    # PyDub
    try:
        from pydub import AudioSegment
        print("✅ PyDub importado correctamente")
    except ImportError:
        AudioSegment = None
        missing_deps.append("pydub")

    # NLTK
    try:
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        print("✅ NLTK importado correctamente")
    except ImportError:
        nltk = None
        sent_tokenize = None
        word_tokenize = None
        stopwords = None
        missing_deps.append("nltk")

    # Estándar
    try:
        from collections import Counter
        import numpy as np
        print("✅ Dependencias estándar importadas correctamente")
    except ImportError as e:
        print(f"❌ Error con dependencias estándar: {e}")
        Counter = None  # type: ignore
        np = None  # type: ignore
        missing_deps.append("numpy")

    if missing_deps:
        print(f"\n❌ Faltan dependencias: {', '.join(missing_deps)}")
        print("🔧 Instala con: pip install -r requirements.txt")
        sys.exit(1)

    # Devolvemos placeholders compatibles con el código previo
    torch = None  # Ya no es necesario
    whisper = WhisperModel  # Referencia a la clase para mantener orden de retorno
    return whisper, torch, VideoFileClip, AudioSegment, nltk, sent_tokenize, word_tokenize, stopwords, Counter, np

# Verificar e importar dependencias SOLO en procesos hijos o si se fuerza explícitamente.
# Evitar importar librerías nativas en el proceso padre interactivo para que no
# arranquen hilos/recursos que luego provoquen segfaults al mezclarse con otras
# extensiones nativas. El child se lanzará con --child-run por la lógica del
# script, por lo que aquí detectamos ese flag.
if '--child-run' in sys.argv or os.environ.get('FORCE_IMPORT_DEPENDENCIES') == '1':
    whisper, torch, VideoFileClip, AudioSegment, nltk, sent_tokenize, word_tokenize, stopwords, Counter, np = check_and_import_dependencies()
else:
    # Placeholders para no importar dependencias en el proceso padre
    whisper = None
    torch = None
    VideoFileClip = None
    AudioSegment = None
    nltk = None
    sent_tokenize = None
    word_tokenize = None
    stopwords = None
    Counter = None
    np = None

# Extensiones de archivo soportadas (usadas en selección y procesamiento)
video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma'}

class AudioTranscriberSummarizer:
    def __init__(self, model_size="base", language="auto"):
        """Inicializa el transcriptor y resumidor usando faster-whisper.

        Args:
            model_size: 'tiny', 'base', 'small', 'medium', 'large-v2', etc.
            language: Código de idioma o 'auto'.
        """
        from faster_whisper import WhisperModel
        print("🤖 Inicializando Audio Transcriber & Summarizer (faster-whisper)...")
        print(f"📡 Cargando modelo '{model_size}' ...")

        # Intentar configuraciones preferidas de precisión
        # En Apple Silicon suele funcionar float16; fallback a int8 para menos RAM.
        preferred_compute_types = ["float16", "int8_float16", "int8"]
        model_loaded = False
        last_error = None
        for compute_type in preferred_compute_types:
            try:
                self.model = WhisperModel(model_size, device="auto", compute_type=compute_type)
                print(f"✅ Modelo cargado con compute_type={compute_type}")
                model_loaded = True
                break
            except Exception as e:  # noqa: BLE001
                last_error = e
                continue
        if not model_loaded:
            raise RuntimeError(f"No se pudo cargar el modelo {model_size}: {last_error}")

        self.language = language
        # Timeout (s) for potentially hanging transcribe calls
        self.transcribe_timeout = 300
        self._setup_nltk()
        print("✅ Inicialización completada")
    
    def _setup_nltk(self):
        """Configura las dependencias de NLTK"""
        try:
            # Descargar recursos necesarios de NLTK
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            print(f"⚠️  Advertencia: Error configurando NLTK: {e}")
    
    def extract_audio_from_video(self, video_path: str, audio_path: Optional[str] = None) -> str:
        """
        Extrae audio de un archivo de video
        
        Args:
            video_path (str): Ruta al archivo de video
            audio_path (str): Ruta donde guardar el audio (opcional)
        
        Returns:
            str: Ruta del archivo de audio extraído
        """
        if audio_path is None:
            video_name = Path(video_path).stem
            audio_path = f"{video_name}_audio.wav"
        
        print(f"🎬 Extrayendo audio de: {os.path.basename(video_path)}")
        
        try:
            # Usando MoviePy para extraer audio
            video = VideoFileClip(video_path)
            audio = video.audio
            # Usar método más simple y compatible
            audio.write_audiofile(audio_path)
            audio.close()
            video.close()
            
            print(f"🎵 Audio extraído: {os.path.basename(audio_path)}")
            return audio_path
            
        except Exception as e:
            print(f"❌ Error al extraer audio: {e}")
            return None
    
    def extract_audio_from_audio(self, input_audio_path: str, output_audio_path: Optional[str] = None) -> str:
        """
        Procesa un archivo de audio (útil para conversión de formato)
        
        Args:
            input_audio_path (str): Ruta al archivo de audio de entrada
            output_audio_path (str): Ruta donde guardar el audio procesado
        
        Returns:
            str: Ruta del archivo de audio procesado
        """
        if output_audio_path is None:
            audio_name = Path(input_audio_path).stem
            output_audio_path = f"{audio_name}_processed.wav"
        
        print(f"🎵 Procesando audio: {os.path.basename(input_audio_path)}")
        
        try:
            # Usar pydub para procesar audio
            audio = AudioSegment.from_file(input_audio_path)
            audio.export(output_audio_path, format="wav")
            
            print(f"✅ Audio procesado: {os.path.basename(output_audio_path)}")
            return output_audio_path
            
        except Exception as e:
            print(f"❌ Error al procesar audio: {e}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe un archivo de audio usando faster-whisper.

        Devuelve un diccionario similar al formato anterior para minimizar cambios.
        """
        print(f"🎤 Transcribiendo audio: {os.path.basename(audio_path)}")
        t_start = time.time()

        # Simple status prints (avoid background threads that may interact
        # badly with native extensions during cleanup / shutdown).
        print("⏳ Transcribiendo... (progreso en consola)")

        try:
            # Parámetros de transcripción
            transcribe_kwargs = {
                "vad_filter": True,
                "beam_size": 5,
            }
            if self.language != "auto":
                transcribe_kwargs["language"] = self.language

            print("⏱️ Llamando a model.transcribe() con argumentos:", transcribe_kwargs)
            t_trans_start = time.time()

            # Ejecutar la llamada potencialmente bloqueante en un hilo para aplicar timeout
            call_result = {}

            def _call_transcribe():
                try:
                    call_result['value'] = self.model.transcribe(audio_path, **transcribe_kwargs)
                except Exception as e:
                    call_result['exc'] = e

            trans_thread = threading.Thread(target=_call_transcribe, daemon=True)
            trans_thread.start()
            trans_thread.join(timeout=self.transcribe_timeout)

            if trans_thread.is_alive():
                print(f"\n⏳ Tiempo de espera excedido ({self.transcribe_timeout}s) en model.transcribe().")
                print("⚠️ El proceso de transcripción parece haberse quedado colgado. Prueba ejecutar con --reinit-each para reinicializar el modelo entre archivos, o usa un modelo más pequeño (tiny) para diagnosticar.)")
                # No podemos forzar la terminación del hilo de forma segura; informar y salir
                return None

            if 'exc' in call_result:
                raise call_result['exc']

            segments_iter, info = call_result.get('value', (None, None))
            print(f"⏱️ model.transcribe() llamada completada en {time.time()-t_trans_start:.2f}s, comenzando a iterar segmentos...")
            segments = []
            full_text_parts = []
            for i, seg in enumerate(segments_iter):
                seg_text = seg.text.strip()
                # provide inline feedback for each segment
                try:
                    print(f"\n  • Segment {i}: {seg.start:.2f}s - {seg.end:.2f}s -> {seg_text[:120]}")
                except Exception:
                    pass
                segments.append({
                    "id": i,
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg_text,
                })
                full_text_parts.append(seg_text)

            full_text = " ".join(full_text_parts).strip()
            result = {
                "text": full_text,
                "language": getattr(info, 'language', None) or "desconocido",
                "duration": float(getattr(info, "duration", 0.0)),
                "segments": segments,
            }

            elapsed = time.time() - t_start
            print("\n✅ Transcripción completada!")
            print(f"🌍 Idioma detectado: {result.get('language', 'desconocido')}")
            print(f"📝 Longitud del texto: {len(result['text'])} caracteres")
            print(f"⏱️ Tiempo total transcripción (incluyendo iteración segmentos): {elapsed:.2f}s")
            return result
        except Exception as e:  # noqa: BLE001
            print(f"\n❌ Error al transcribir: {e}")
            return None
        finally:
            # Forzar recolección de basura y limpieza de caché CUDA si está disponible
            try:
                gc.collect()
            except Exception:
                pass
            try:
                try:
                    import torch as _torch
                except Exception:
                    _torch = None

                if _torch is not None and hasattr(_torch, 'cuda'):
                    try:
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                            print('♻️ CUDA cache cleared')
                    except Exception:
                        pass
            except Exception:
                # No torch installed or cannot clear cache
                pass
    
    def clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto transcrito"""
        # Eliminar espacios extra
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Corregir puntuación común
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*([a-zA-Z])', r'\1 \2', text)
        
        return text
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """
        Extrae palabras clave del texto
        
        Args:
            text (str): Texto a analizar
            num_keywords (int): Número de palabras clave a extraer
        
        Returns:
            list: Lista de palabras clave
        """
        try:
            # Tokenizar palabras
            words = word_tokenize(text.lower())
            
            # Obtener stopwords en español e inglés
            try:
                stop_words_es = set(stopwords.words('spanish'))
                stop_words_en = set(stopwords.words('english'))
                stop_words = stop_words_es.union(stop_words_en)
            except:
                stop_words = set()
            
            # Filtrar palabras
            filtered_words = [
                word for word in words 
                if word.isalpha() and len(word) > 3 and word not in stop_words
            ]
            
            # Contar frecuencias
            word_freq = Counter(filtered_words)
            
            # Obtener las más frecuentes
            keywords = [word for word, freq in word_freq.most_common(num_keywords)]
            
            return keywords
            
        except Exception as e:
            print(f"⚠️  Error extrayendo palabras clave: {e}")
            return []
    
    def create_extractive_summary(self, text: str, num_sentences: int = 5) -> str:
        """
        Crea un resumen extractivo basado en las oraciones más importantes
        
        Args:
            text (str): Texto a resumir
            num_sentences (int): Número de oraciones en el resumen
        
        Returns:
            str: Resumen extractivo
        """
        try:
            # Tokenizar oraciones
            sentences = sent_tokenize(text)
            
            if len(sentences) <= num_sentences:
                return text
            
            # Tokenizar palabras para análisis de frecuencia
            words = word_tokenize(text.lower())
            
            # Obtener stopwords
            try:
                stop_words_es = set(stopwords.words('spanish'))
                stop_words_en = set(stopwords.words('english'))
                stop_words = stop_words_es.union(stop_words_en)
            except:
                stop_words = set()
            
            # Calcular frecuencia de palabras
            filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
            word_freq = Counter(filtered_words)
            
            # Puntuar oraciones basándose en frecuencia de palabras
            sentence_scores = {}
            for sentence in sentences:
                sentence_words = word_tokenize(sentence.lower())
                score = 0
                word_count = 0
                
                for word in sentence_words:
                    if word in word_freq:
                        score += word_freq[word]
                        word_count += 1
                
                if word_count > 0:
                    sentence_scores[sentence] = score / word_count
            
            # Seleccionar las mejores oraciones
            best_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            summary_sentences = [sent for sent, score in best_sentences[:num_sentences]]
            
            # Mantener el orden original
            ordered_summary = []
            for sentence in sentences:
                if sentence in summary_sentences:
                    ordered_summary.append(sentence)
            
            return ' '.join(ordered_summary)
            
        except Exception as e:
            print(f"⚠️  Error creando resumen extractivo: {e}")
            # Resumen simple como fallback
            sentences = text.split('. ')
            if len(sentences) <= num_sentences:
                return text
            return '. '.join(sentences[:num_sentences]) + '.'
    
    def create_topic_summary(self, text: str) -> Dict[str, str]:
        """
        Crea un resumen organizado por temas principales
        
        Args:
            text (str): Texto a analizar
        
        Returns:
            dict: Resumen organizado por temas
        """
        try:
            # Dividir en párrafos
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            if not paragraphs:
                # Si no hay párrafos, dividir por oraciones largas
                sentences = sent_tokenize(text)
                # Agrupar oraciones en "párrafos" de 3-5 oraciones
                paragraphs = []
                for i in range(0, len(sentences), 4):
                    paragraph = ' '.join(sentences[i:i+4])
                    paragraphs.append(paragraph)
            
            topics = {}
            
            for i, paragraph in enumerate(paragraphs[:5]):  # Limitar a 5 temas principales
                # Extraer palabras clave del párrafo
                keywords = self.extract_keywords(paragraph, num_keywords=3)
                
                # Crear nombre del tema
                if keywords:
                    topic_name = f"Tema {i+1}: {', '.join(keywords[:2])}"
                else:
                    topic_name = f"Tema {i+1}"
                
                # Resumir el párrafo
                summary = self.create_extractive_summary(paragraph, num_sentences=2)
                topics[topic_name] = summary
            
            return topics
            
        except Exception as e:
            print(f"⚠️  Error creando resumen por temas: {e}")
            return {"Resumen general": self.create_extractive_summary(text)}
    
    def process_media_file(self, file_path: str, output_dir: Optional[str] = None, 
                          keep_audio: bool = False, summary_sentences: int = 5) -> Dict:
        """
        Procesa un archivo de medios (video o audio) completo
        
        Args:
            file_path (str): Ruta al archivo de medios
            output_dir (str): Directorio de salida (por defecto: carpeta transcripciones)
            keep_audio (bool): Mantener archivo de audio extraído
            summary_sentences (int): Número de oraciones en el resumen
        
        Returns:
            dict: Resultado completo del procesamiento
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ Error: El archivo {file_path} no existe")
            return None
        
        # Configurar directorios
        if output_dir is None:
            # Usar carpeta transcripciones por defecto
            output_dir = Path("/Users/Alexis/Desktop/githubroot/github/youtube/transcripciones")
        else:
            output_dir = Path(output_dir)
        
        # Crear directorio si no existe
        output_dir.mkdir(exist_ok=True)
        
        print(f"{'='*40}")
        print("� INICIANDO TRANSCRIPCIÓN")
        print(f"{'='*40}")
        
        file_extension = file_path.suffix.lower()
        
        # Extraer o procesar audio
        temp_audio_path = None
        if file_extension in video_extensions:
            # Es un video, extraer audio
            temp_audio_path = output_dir / f"{file_path.stem}_temp_audio.wav"
            audio_path = self.extract_audio_from_video(str(file_path), str(temp_audio_path))
        elif file_extension in audio_extensions:
            # Es un audio, usar directamente o procesar si es necesario
            if file_extension == '.wav':
                audio_path = str(file_path)
            else:
                temp_audio_path = output_dir / f"{file_path.stem}_temp_audio.wav"
                audio_path = self.extract_audio_from_audio(str(file_path), str(temp_audio_path))
        else:
            print(f"❌ Error: Formato de archivo no soportado: {file_extension}")
            return None
        
        if not audio_path:
            print("❌ Error: No se pudo extraer/procesar el audio")
            return None
        
        # Transcribir
        print(f"\n{'='*40}")
        print("🎤 INICIANDO TRANSCRIPCIÓN")
        print(f"{'='*40}")
        
        print("⏱️ Llamando a transcribe_audio()...")
        sys.stdout.flush()
        transcription = self.transcribe_audio(audio_path)
        print("⏱️ transcribe_audio() finalizó")
        sys.stdout.flush()

        if not transcription:
            print("❌ Error: No se pudo transcribir el audio")
            return None
        
        # Limpiar texto
        print("⏱️ Limpiando texto transcrito...")
        sys.stdout.flush()
        clean_text = self.clean_text(transcription["text"])

        # Helper para ejecutar funciones con timeout en hilo
        def _run_with_timeout(fn, args=(), kwargs=None, timeout=120):
            kwargs = kwargs or {}
            result = {}

            def _target():
                try:
                    result['value'] = fn(*args, **kwargs)
                except Exception as e:
                    result['exc'] = e

            th = threading.Thread(target=_target, daemon=True)
            th.start()
            th.join(timeout=timeout)
            if th.is_alive():
                return {'timeout': True}
            if 'exc' in result:
                return {'exc': result['exc']}
            return {'value': result.get('value')}

        # Extraer palabras clave
        print(f"\n🔍 Extrayendo palabras clave...")
        sys.stdout.flush()
        kw_res = _run_with_timeout(self.extract_keywords, args=(clean_text,), timeout=60)
        if 'timeout' in kw_res:
            print("⚠️ Timeout en extracción de palabras clave (60s). Se continuará sin keywords.")
            keywords = []
        elif 'exc' in kw_res:
            print(f"⚠️ Error en extract_keywords: {kw_res['exc']}")
            keywords = []
        else:
            keywords = kw_res.get('value', []) or []

        # Crear resúmenes
        print(f"📋 Creando resúmenes...")
        sys.stdout.flush()
        ex_res = _run_with_timeout(self.create_extractive_summary, args=(clean_text, summary_sentences), timeout=90)
        if 'timeout' in ex_res:
            print("⚠️ Timeout en resumen extractivo (90s). Usando texto original como fallback.")
            extractive_summary = clean_text[:1000]
        elif 'exc' in ex_res:
            print(f"⚠️ Error en create_extractive_summary: {ex_res['exc']}")
            extractive_summary = clean_text[:1000]
        else:
            extractive_summary = ex_res.get('value', '') or ''

        topic_res = _run_with_timeout(self.create_topic_summary, args=(clean_text,), timeout=90)
        if 'timeout' in topic_res:
            print("⚠️ Timeout en resumen por temas (90s). Creando resumen general fallback.")
            topic_summary = {"Resumen general": extractive_summary}
        elif 'exc' in topic_res:
            print(f"⚠️ Error en create_topic_summary: {topic_res['exc']}")
            topic_summary = {"Resumen general": extractive_summary}
        else:
            topic_summary = topic_res.get('value', {}) or {"Resumen general": extractive_summary}
        
        # Preparar resultado completo
        result = {
            "file_info": {
                "original_file": str(file_path),
                "file_type": "video" if file_extension in video_extensions else "audio",
                "processed_date": datetime.now().isoformat(),
                "file_size_mb": round(file_path.stat().st_size / (1024*1024), 2)
            },
            "transcription": {
                "language": transcription.get("language", "unknown"),
                "duration_seconds": transcription.get("duration", 0),
                "full_text": clean_text,
                "raw_text": transcription["text"],
                "segments": transcription.get("segments", [])
            },
            "analysis": {
                "text_stats": {
                    "character_count": len(clean_text),
                    "word_count": len(clean_text.split()),
                    "sentence_count": len(sent_tokenize(clean_text))
                },
                "keywords": keywords,
                "extractive_summary": extractive_summary,
                "topic_summary": topic_summary
            }
        }
        
        # Guardar resultados
        print(f"\n💾 Guardando resultados...")
        
        # Archivo JSON completo
        json_file = output_dir / f"{file_path.stem}_analysis.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"📄 Análisis completo: {json_file.name}")
        
        # Archivo de texto con transcripción
        text_file = output_dir / f"{file_path.stem}_transcription.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(clean_text)
        print(f"📝 Transcripción: {text_file.name}")
        
        # Archivo de resumen
        summary_file = output_dir / f"{file_path.stem}_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("RESUMEN EXTRACTIVO\n")
            f.write("="*50 + "\n\n")
            f.write(extractive_summary)
            f.write("\n\n")
            
            f.write("PALABRAS CLAVE\n")
            f.write("="*50 + "\n")
            f.write(", ".join(keywords))
            f.write("\n\n")
            
            f.write("RESUMEN POR TEMAS\n")
            f.write("="*50 + "\n")
            for topic, summary in topic_summary.items():
                f.write(f"\n{topic}:\n")
                f.write("-" * len(topic) + "\n")
                f.write(summary)
                f.write("\n")
        
        print(f"📋 Resumen: {summary_file.name}")
        
        # Limpiar archivo temporal de audio
        if temp_audio_path and not keep_audio and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print("🗑️  Archivo de audio temporal eliminado")
        elif temp_audio_path and keep_audio:
            final_audio_path = output_dir / f"{file_path.stem}_audio.wav"
            if temp_audio_path != final_audio_path:
                os.rename(temp_audio_path, final_audio_path)
                print(f"🎵 Audio guardado: {final_audio_path.name}")
        
        return result

def select_video_file():
    """
    Permite al usuario seleccionar un archivo de video de la carpeta videos
    
    Returns:
        str: Ruta completa al archivo seleccionado o None si se cancela
    """
    videos_dir = "/Users/Alexis/Desktop/githubroot/github/youtube/videos"
    
    # Verificar que la carpeta existe
    if not os.path.exists(videos_dir):
        print(f"❌ Error: La carpeta {videos_dir} no existe")
        return None
    
    # Obtener lista de archivos de video
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    video_files = []
    
    try:
        for file in os.listdir(videos_dir):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(file)
    except Exception as e:
        print(f"❌ Error al leer la carpeta: {e}")
        return None
    
    if not video_files:
        print("❌ No se encontraron archivos de video en la carpeta")
        return None
    
    # Ordenar archivos alfabéticamente
    video_files.sort()
    
    # Mostrar menú de selección
    print(f"\n🎬 ARCHIVOS DE VIDEO DISPONIBLES")
    print(f"{'='*60}")
    print(f"📁 Carpeta: {videos_dir}")
    print(f"📊 Total de videos: {len(video_files)}")
    print(f"{'='*60}")
    
    for i, file in enumerate(video_files, 1):
        file_path = os.path.join(videos_dir, file)
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"{i:2d}. {file} ({file_size:.1f} MB)")
    
    print(f"\n0. ❌ Cancelar")
    print(f"{'='*60}")
    
    # Solicitar selección
    while True:
        try:
            choice = input(f"🎯 Selecciona un archivo (1-{len(video_files)}) o 0 para cancelar: ").strip()
            
            if choice == '0':
                print("❌ Operación cancelada")
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(video_files):
                selected_file = video_files[choice_num - 1]
                full_path = os.path.join(videos_dir, selected_file)
                print(f"\n✅ Archivo seleccionado: {selected_file}")
                return full_path
            else:
                print(f"❌ Por favor, ingresa un número entre 1 y {len(video_files)}")
                
        except ValueError:
            print("❌ Por favor, ingresa un número válido")
        except KeyboardInterrupt:
            print("\n❌ Operación cancelada")
            return None

def main():
    parser = argparse.ArgumentParser(
        description='🎯 Audio Transcriber & Summarizer - Extrae audio, transcribe y resume contenido'
    )
    
    parser.add_argument('file_path', nargs='?', help='Ruta al archivo de video o audio (opcional)')
    parser.add_argument('--model', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Tamaño del modelo Whisper (default: base)')
    parser.add_argument('--language', default='auto',
                       help='Código de idioma (ej: es, en, fr) o "auto" para detección automática')
    parser.add_argument('--output-dir', help='Directorio de salida')
    parser.add_argument('--keep-audio', action='store_true',
                       help='Mantener archivo de audio extraído')
    parser.add_argument('--summary-sentences', type=int, default=5,
                       help='Número de oraciones en el resumen extractivo (default: 5)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Modo interactivo para seleccionar archivo de video')
    parser.add_argument('--reinit-each', action='store_true',
                       help='Reinicializar el modelo entre cada archivo (evita acumulación de estado; más lento)')
    parser.add_argument('--isolated', action='store_true',
                       help='Ejecutar cada transcripción en un subproceso aislado (más seguro, más lento)')
    parser.add_argument('--child-run', action='store_true',
                       help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    def run_file_in_subprocess(file_path: str, parsed_args, timeout: int = 1200):
        """Lanza un subproceso Python aislado para procesar un único archivo.

        El subproceso ejecuta este mismo script con --child-run y los parámetros necesarios.
        """
        script_path = os.path.abspath(__file__)
        cmd = [sys.executable, script_path, file_path, '--child-run', '--model', parsed_args.model, '--language', parsed_args.language]
        if parsed_args.output_dir:
            cmd += ['--output-dir', parsed_args.output_dir]
        if parsed_args.keep_audio:
            cmd += ['--keep-audio']
        if parsed_args.summary_sentences:
            cmd += ['--summary-sentences', str(parsed_args.summary_sentences)]
        # Keep reinit flag in child if requested
        if getattr(parsed_args, 'reinit_each', False):
            cmd += ['--reinit-each']

        print(f"🔁 Ejecutando archivo en subproceso: {' '.join(cmd)}")
        try:
            # Prepare a reduced environment for the child to avoid MKL/OpenMP
            # thread clashes and to limit native parallelism which often helps
            # avoid segfaults in mixed native-extension workloads.
            env = os.environ.copy()
            env.update({
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1',
                # Ensure tqdm monitor threads are discouraged in child too
                'TQDM_DISABLE': env.get('TQDM_DISABLE', '1'),
                # Keep existing KMP duplicate setting
                'KMP_DUPLICATE_LIB_OK': env.get('KMP_DUPLICATE_LIB_OK', 'TRUE'),
            })

            # Run child in a new session so signals and subprocess cleanup are isolated.
            result = subprocess.run(cmd, check=False, timeout=timeout, env=env, start_new_session=True, close_fds=True)
            print(f"🔚 Subproceso finalizado con código de salida: {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"⏳ Tiempo de espera ({timeout}s) agotado para el subproceso. Se abortó la transcripción.")
        except Exception as e:
            print(f"❌ Error ejecutando subproceso: {e}")

    # Si se invocó como child-run, procesar solo el archivo y salir
    if args.child_run:
        if not args.file_path:
            print("❌ child-run requiere una ruta de archivo como argumento")
            sys.exit(2)
        # Procesar el archivo en este proceso hijo
        processor = None
        def run_and_exit(file_path_arg):
            nonlocal processor
            if processor is None:
                try:
                    processor = AudioTranscriberSummarizer(args.model, args.language)
                except Exception as e:
                    print(f"❌ Error inicializando el procesador en child: {e}")
                    return 2
            res = processor.process_media_file(file_path_arg, args.output_dir, args.keep_audio, args.summary_sentences)
            return 0 if res else 3

        code = run_and_exit(args.file_path)
        sys.exit(code)
    
    # Determinar modo: interactivo o archivo puntual
    processor = None  # lazy init

    def run_and_show(file_path):
        nonlocal processor
        # Crear procesador si no existe (carga el modelo la primera vez)
        if processor is None:
            try:
                processor = AudioTranscriberSummarizer(args.model, args.language)
            except Exception as e:
                print(f"❌ Error inicializando el procesador: {e}")
                return None

        # Verificar archivo
        if not os.path.exists(file_path):
            print(f"❌ Error: El archivo {file_path} no existe")
            return None

        # Procesar archivo
        result = processor.process_media_file(
            file_path,
            args.output_dir,
            args.keep_audio,
            args.summary_sentences
        )

        if not result:
            print("❌ Error: No se pudo completar el procesamiento")
            return None

        # Mostrar resumen/estadísticas (igual que antes)
        print(f"\n{'='*60}")
        print("✅ PROCESAMIENTO COMPLETADO")
        print(f"{'='*60}")

        stats = result["analysis"]["text_stats"]
        print(f"📊 Estadísticas:")
        print(f"   • Idioma: {result['transcription']['language']}")
        print(f"   • Duración: {result['transcription']['duration_seconds']:.1f} segundos")
        print(f"   • Palabras: {stats['word_count']}")
        print(f"   • Oraciones: {stats['sentence_count']}")
        print(f"   • Caracteres: {stats['character_count']}")

        print(f"\n🔑 Palabras clave:")
        print(f"   {', '.join(result['analysis']['keywords'][:8])}")

        print(f"\n📝 TRANSCRIPCIÓN COMPLETA:")
        print(f"{'='*60}")
        print(result["transcription"]["full_text"])
        print(f"{'='*60}")

        return result

    # Interactive loop: if interactive mode or no file provided, keep showing menu after each transcription
    if args.interactive or not args.file_path:
        print("🎯 Modo interactivo activado — el menú volverá después de cada transcripción")
        while True:
            file_path = select_video_file()
            if not file_path:
                print("❌ Operación cancelada. Saliendo.")
                break

            # To avoid native crashes leaking into the interactive loop
            # (segfaults from native extension threads), always run the
            # actual processing in an isolated subprocess. This confines
            # crashes to the child and keeps the parent interactive shell
            # responsive. The existing --isolated flag is still supported
            # but we force isolation here for safety.
            run_file_in_subprocess(file_path, args)

            # Flush prints and short sleep to ensure terminal state is stable before prompting
            try:
                sys.stdout.flush()
            except Exception:
                pass
            time.sleep(0.05)

            # Si se solicita, reinicializar el procesador para forzar recarga del modelo
            if args.reinit_each and processor is not None:
                try:
                    print("🔁 Reinicializando el modelo para la próxima corrida (liberando memoria)")
                    del processor
                    processor = None
                    try:
                        gc.collect()
                    except Exception:
                        pass
                    try:
                        import torch as _torch
                        if hasattr(_torch, 'cuda') and _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                    except Exception:
                        pass
                except Exception:
                    pass

            # Al finalizar, preguntar si desea procesar otro (volver al menú)
            try:
                # Best-effort: stop tqdm monitor threads before blocking on input to
                # avoid segfaults caused by lingering monitor threads interacting
                # with native extensions.
                _stop_tqdm_monitor()
                cont = input("\n¿Procesar otro archivo? [Y/n]: ").strip().lower()
            except KeyboardInterrupt:
                print("\n❌ Interrumpido por usuario. Saliendo.")
                break
            if cont and cont.startswith('n'):
                print("Saliendo del modo interactivo.")
                break

        # Fin del modo interactivo
    else:
        # Modo no interactivo: procesar el archivo proporcionado y salir
        file_path = args.file_path
        if not file_path or not os.path.exists(file_path):
            print(f"❌ Error: El archivo {file_path} no existe")
            sys.exit(1)

        # Always run processing in an isolated subprocess to confine any
        # native crashes (from libs like av, torch, numpy MKL, etc.) to the
        # child process. Users can still opt-out by adding a new flag later,
        # but defaulting to isolation is the safest behavior on macOS/conda.
        run_file_in_subprocess(file_path, args)
        result = None
        if result is None:
            sys.exit(1)
        # Reinicializar si se solicitó (no-interactivo)
        if args.reinit_each and processor is not None:
            try:
                del processor
                processor = None
                try:
                    gc.collect()
                except Exception:
                    pass
                try:
                    import torch as _torch
                    if hasattr(_torch, 'cuda') and _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception:
                pass

if __name__ == "__main__":
    main()
