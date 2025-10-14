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
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import re
from typing import List, Dict, Optional

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

# Verificar e importar dependencias
whisper, torch, VideoFileClip, AudioSegment, nltk, sent_tokenize, word_tokenize, stopwords, Counter, np = check_and_import_dependencies()

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

        try:
            # Parámetros de transcripción
            transcribe_kwargs = {
                "vad_filter": True,
                "beam_size": 5,
            }
            if self.language != "auto":
                transcribe_kwargs["language"] = self.language

            segments_iter, info = self.model.transcribe(audio_path, **transcribe_kwargs)
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
            result = {
                "text": full_text,
                "language": info.language or "desconocido",
                "duration": getattr(info, "duration", 0.0),
                "segments": segments,
            }

            print("✅ Transcripción completada!")
            print(f"🌍 Idioma detectado: {result.get('language', 'desconocido')}")
            print(f"📝 Longitud del texto: {len(result['text'])} caracteres")
            return result
        except Exception as e:  # noqa: BLE001
            print(f"❌ Error al transcribir: {e}")
            return None
    
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
        
        print(f"\n{'='*60}")
        print(f"🎯 PROCESANDO: {file_path.name}")
        print(f"📁 GUARDANDO EN: {output_dir}")
        print(f"{'='*60}")
        
        # Determinar si es video o audio
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
        
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
        
        transcription = self.transcribe_audio(audio_path)
        
        if not transcription:
            print("❌ Error: No se pudo transcribir el audio")
            return None
        
        # Limpiar texto
        clean_text = self.clean_text(transcription["text"])
        
        # Extraer palabras clave
        print(f"\n🔍 Extrayendo palabras clave...")
        keywords = self.extract_keywords(clean_text)
        
        # Crear resúmenes
        print(f"📋 Creando resúmenes...")
        extractive_summary = self.create_extractive_summary(clean_text, summary_sentences)
        topic_summary = self.create_topic_summary(clean_text)
        
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
    
    args = parser.parse_args()
    
    # Determinar archivo a procesar
    file_path = None
    
    if args.interactive or not args.file_path:
        # Modo interactivo o no se proporcionó archivo
        print("🎯 Modo de selección interactiva activado")
        file_path = select_video_file()
        if not file_path:
            sys.exit(1)
    else:
        file_path = args.file_path
    
    # Verificar archivo
    if not os.path.exists(file_path):
        print(f"❌ Error: El archivo {file_path} no existe")
        sys.exit(1)
    
    # Crear procesador
    try:
        processor = AudioTranscriberSummarizer(args.model, args.language)
    except Exception as e:
        print(f"❌ Error inicializando el procesador: {e}")
        sys.exit(1)
    
    # Procesar archivo
    result = processor.process_media_file(
        file_path,
        args.output_dir,
        args.keep_audio,
        args.summary_sentences
    )
    
    if result:
        print(f"\n{'='*60}")
        print("✅ PROCESAMIENTO COMPLETADO")
        print(f"{'='*60}")
        
        # Mostrar estadísticas
        stats = result["analysis"]["text_stats"]
        print(f"📊 Estadísticas:")
        print(f"   • Idioma: {result['transcription']['language']}")
        print(f"   • Duración: {result['transcription']['duration_seconds']:.1f} segundos")
        print(f"   • Palabras: {stats['word_count']}")
        print(f"   • Oraciones: {stats['sentence_count']}")
        print(f"   • Caracteres: {stats['character_count']}")
        
        # Mostrar palabras clave
        print(f"\n🔑 Palabras clave:")
        print(f"   {', '.join(result['analysis']['keywords'][:8])}")
        
        # Mostrar transcripción completa
        print(f"\n📝 TRANSCRIPCIÓN COMPLETA:")
        print(f"{'='*60}")
        print(result["transcription"]["full_text"])
        print(f"{'='*60}")
        
    else:
        print("❌ Error: No se pudo completar el procesamiento")
        sys.exit(1)

if __name__ == "__main__":
    main()
