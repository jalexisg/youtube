# Audio Transcriber & Summarizer (Web & CLI)

Un proyecto Python avanzado para extraer audio de videos, transcribir el contenido usando Whisper de OpenAI y generar resúmenes inteligentes automáticamente. Ahora con una **Interfaz Web Moderna**.

## 🚀 Características Principales

- 🌐 **Interfaz Web Moderna**: Sube archivos o pega links de Instagram y YouTube cómodamente.
- 🐳 **Docker Ready**: Despliegue fácil y rápido sin preocuparse por dependencias.
- 📥 **Descargas**: Descarga el video/audio original y el audio procesado.
- 🎥 **YouTube & Shorts**: Soporte completo para videos y Shorts de YouTube.
- 🔐 **Autenticación**: Soporte para videos con restricción de edad (usando `cookies.txt`).
- 🎤 **Transcripción Potente**: Usa Whisper de OpenAI (versión `faster-whisper`).
- 📝 **Resúmenes Inteligentes**: Genera resúmenes extractivos y por temas.
- 🌍 **Multi-idioma**: Detección automática de idioma.

## 📦 Instalación y Uso (Recomendado: Docker)

La forma más sencilla de usar la aplicación es con Docker.

### Requisitos
- Docker
- Docker Compose
- (Opcional) `cookies.txt` en la raíz del proyecto para videos restringidos de YouTube.

### Pasos

1.  **Clonar el repositorio**:
    ```bash
    git clone <url-del-repo>
    cd youtube
    ```

2.  **Iniciar la aplicación**:
    ```bash
    docker-compose up --build
    ```

3.  **Usar la Web**:
    Abre tu navegador en **[http://localhost:8000](http://localhost:8000)**.

4.  **Detener**:
    Presiona `Ctrl+C` en la terminal.

---

## 🔧 Instalación Manual (Local)

Si prefieres ejecutarlo sin Docker:

### Requisitos previos
1.  **Python 3.8+**
2.  **FFmpeg**:
    - macOS: `brew install ffmpeg`
    - Linux: `sudo apt install ffmpeg`

### Pasos

1.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Iniciar el servidor web**:
    ```bash
    python web_server.py
    ```

3.  **Acceder**:
    Ve a [http://localhost:8000](http://localhost:8000).

---

## 💻 Uso por Línea de Comandos (CLI)

También puedes usar los scripts directamente desde la terminal.

### Script Principal: `audio_transcriber_summarizer.py`

#### Modo Interactivo
```bash
python audio_transcriber_summarizer.py --interactive
```

#### Uso Básico
```bash
# Archivo local
python audio_transcriber_summarizer.py video.mp4

# Opciones avanzadas
python audio_transcriber_summarizer.py video.mp4 --model medium --language es --summary-sentences 10
```

## 📄 Archivos de Salida

El sistema genera los siguientes archivos en la carpeta `transcripciones/` (o descargables desde la web):

1.  **`*_analysis.json`**: Datos completos, transcripción, estadísticas y palabras clave.
2.  **`*_transcription.txt`**: Texto plano de la transcripción.
3.  **`*_summary.txt`**: Resumen extractivo y por temas.
4.  **`*_audio.wav`**: Audio procesado (opcional).

## 🔧 Estructura del Proyecto

```
youtube/
├── web_server.py            # Backend FastAPI
├── audio_transcriber_summarizer.py # Core logic
├── Dockerfile              # Configuración Docker
├── docker-compose.yml      # Orquestación Docker
├── requirements.txt        # Dependencias
├── cookies.txt             # (Opcional) Cookies de YouTube
├── web/                    # Frontend (HTML/CSS/JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── videos/                 # Carpeta para videos descargados
└── transcripciones/        # Resultados generados
```

## Modelos de Whisper

| Modelo | Tamaño | Velocidad | Precisión | Uso recomendado |
|--------|--------|-----------|-----------|-----------------|
| tiny   | ~39 MB | Muy rápido | Básica | Pruebas rápidas |
| base   | ~74 MB | Rápido | Buena | Uso general |
| small  | ~244 MB | Medio | Muy buena | Balance calidad/velocidad |
| medium | ~769 MB | Lento | Excelente | Alta calidad |
| large  | ~1550 MB | Muy lento | Máxima | Máxima precisión |

## Licencia

MIT License