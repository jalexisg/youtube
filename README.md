# Audio Transcriber & Summarizer

Un proyecto Python avanzado para extraer audio de videos, transcribir el contenido usando Whisper de OpenAI y generar resúmenes inteligentes automáticamente.

## 🚀 Características Principales

- 🎥 **Extrae audio** de archivos de video (MP4, AVI, MOV, MKV, etc.)
- 🎵 **Procesa archivos de audio** directamente (MP3, WAV, FLAC, etc.)
- 🎤 **Transcripción automática** usando Whisper de OpenAI
- 🌍 **Detección automática de idioma** o especificación manual
- 📝 **Resúmenes inteligentes** extractivos y por temas
- 🔍 **Extracción de palabras clave** automática
- 📊 **Estadísticas detalladas** del contenido
- 💾 **Múltiples formatos de salida** (JSON, TXT)
- ⚡ **Soporte para GPU** (CUDA) para procesamiento rápido

## 📦 Instalación

### Requisitos previos

1. **Python 3.8 o superior**
2. **FFmpeg** (para procesamiento de audio/video)

#### Instalar FFmpeg en macOS:
```bash
# Usando Homebrew
brew install ffmpeg

# O usando MacPorts
sudo port install ffmpeg
```

### Instalar dependencias

```bash
# Clonar o descargar el proyecto
cd /Users/Alexis/Desktop/githubroot/github/youtube

# Instalar dependencias de Python
pip install -r requirements.txt
```

## 🎯 Uso

### Script Principal: `audio_transcriber_summarizer.py`

#### Modo Interactivo (Nuevo) 🎯
```bash
# Modo interactivo - selecciona archivo de la carpeta videos
python audio_transcriber_summarizer.py --interactive

# O simplemente (activa automáticamente el modo interactivo)
python audio_transcriber_summarizer.py
```

El modo interactivo te permite:
- 📁 Ver todos los videos disponibles en la carpeta `videos/`
- 📊 Ver el tamaño de cada archivo
- 🎯 Seleccionar fácilmente el archivo que quieres procesar
- ❌ Cancelar la operación si es necesario

#### Uso básico con archivo específico
```bash
python audio_transcriber_summarizer.py archivo.mp4
```

#### Opciones avanzadas
```bash
# Especificar modelo de Whisper
python audio_transcriber_summarizer.py video.mp4 --model medium

# Especificar idioma
python audio_transcriber_summarizer.py video.mp4 --language es

# Mantener archivo de audio extraído
python audio_transcriber_summarizer.py video.mp4 --keep-audio

# Personalizar número de oraciones en el resumen
python audio_transcriber_summarizer.py video.mp4 --summary-sentences 10

# Especificar directorio de salida
python audio_transcriber_summarizer.py video.mp4 --output-dir ./mis_transcripciones/
```

### Script de Ejemplo: `example_usage.py`

```bash
# Ejecutar ejemplos interactivos
python example_usage.py
```

### Ejemplos con tus archivos

```bash
# Transcribir y resumir un video
python audio_transcriber_summarizer.py "pitagoras.mp4" --summary-sentences 5 --language es

# Procesar un archivo de audio
python audio_transcriber_summarizer.py "Por Amor.mp3" --model small --keep-audio

# Procesamiento avanzado con modelo grande
python audio_transcriber_summarizer.py "LA GOTA FRÍA Calixto Acordeón Mágico El Vallenatero.mp4" --model large --language es --output-dir ./resultados/
```

## Modelos de Whisper

| Modelo | Tamaño | Velocidad | Precisión | Uso recomendado |
|--------|--------|-----------|-----------|-----------------|
| tiny   | ~39 MB | Muy rápido | Básica | Pruebas rápidas |
| base   | ~74 MB | Rápido | Buena | Uso general |
| small  | ~244 MB | Medio | Muy buena | Balance calidad/velocidad |
| medium | ~769 MB | Lento | Excelente | Alta calidad |
| large  | ~1550 MB | Muy lento | Máxima | Máxima precisión |

## 📄 Archivos de salida

El nuevo script genera automáticamente varios archivos:

### 1. **`archivo_analysis.json`**
Archivo JSON completo con:
- Información del archivo original
- Transcripción completa con segmentos temporales
- Estadísticas del texto (palabras, oraciones, caracteres)
- Palabras clave extraídas
- Resumen extractivo
- Resumen organizado por temas

### 2. **`archivo_transcription.txt`**
Texto plano limpio de la transcripción completa

### 3. **`archivo_summary.txt`**
Archivo de resumen que incluye:
- Resumen extractivo principal
- Lista de palabras clave
- Resumen organizado por temas principales

### 4. **`archivo_audio.wav`** (opcional)
Archivo de audio extraído (si se especifica `--keep-audio`)

## 🔧 Estructura del proyecto

```
youtube/
├── audio_transcriber_summarizer.py  # Script principal (NUEVO)
├── example_usage.py                 # Ejemplos de uso (NUEVO)
├── video_transcriber.py             # Script original
├── youtubeDownloader.ipynb          # Notebook para descargas
├── download.py                      # Script de descarga
├── image_resizer.py                 # Utilidad para imágenes
├── requirements.txt                 # Dependencias actualizadas
├── README.md                        # Esta documentación
├── src/                            # Código fuente
│   └── youtube_downloader.py
├── tests/                          # Pruebas
│   └── test_youtube_downloader.py
└── transcripciones/                # Directorio de salida (se crea automáticamente)
```

## Códigos de idioma soportados

- `es` - Español
- `en` - Inglés
- `fr` - Francés
- `de` - Alemán
- `it` - Italiano
- `pt` - Portugués
- Y muchos más...

## Solución de problemas

### Error: "ffmpeg not found"
```bash
# En macOS
brew install ffmpeg

# Verificar instalación
ffmpeg -version
```

### Error de memoria insuficiente
- Use un modelo más pequeño (`--model tiny` o `--model base`)
- Cierre otras aplicaciones que consuman memoria

### Audio no se extrae correctamente
- Verifique que el archivo de video no esté corrupto
- Pruebe con otro formato de video

## Licencia

MIT License

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request