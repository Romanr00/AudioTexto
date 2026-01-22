# AudioTexto

Sistema de transcripción inteligente de audio utilizando modelos de lenguaje avanzados (OpenAI GPT-4o-mini / Whisper y Google Gemini 2.5 Flash).

## Características principales
- **Optimización Automática:** Reduce el tamaño de los audios antes de procesarlos para ahorrar ancho de banda y tiempo.
- **Detección de Idiomas:** Extrae información de etiquetas o del nombre del archivo para mejorar la precisión de la transcripción.
- **Multiprocesamiento:** Capacidad de procesar archivos individuales o carpetas completas en lote.
- **Integración con Automator:** Diseñado para funcionar como una acción de clic derecho en macOS.

## Requisitos
- Python 3.10+
- FFmpeg (instalado en el PATH)
- Claves de API de OpenAI y/o Google Gemini.

## Instalación
1. Clona el repositorio.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Configura tus claves en un archivo `.env` (usa `.env.example` como guía).

## Uso
Ejecuta el script principal:
```bash
python3 2_Areas/Python/transcribir_audio.py --file "ruta/al/audio.mp3" --model gemini-2.5-flash
```

O usa la carpeta de entrada configurada:
```bash
python3 2_Areas/Python/transcribir_audio.py --input "2_Areas/Python/input" --model whisper-1
```
