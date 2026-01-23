import os
import sys
import argparse
import logging
import subprocess
import shutil
import re
import time
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import google.generativeai as genai

# Cargar variables de entorno
load_dotenv()

# Configuración de Logging
# Solo archivo para no romper las barras de progreso en consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcription_manager.log")
    ]
)
logger = logging.getLogger(__name__)

# Configuración de Constantes
TARGET_BITRATE = "32k"
TARGET_CHANNELS = "1"  # Mono
TARGET_SAMPLE_RATE = "16000"
CHUNK_TIME_SECONDS = 600  # 10 minutos

def get_openai_client():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: No se encontró OPENAI_API_KEY en las variables de entorno.")
        sys.exit(1)
    return OpenAI(api_key=api_key)

def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg no está instalado o no se encuentra en el PATH.")
        sys.exit(1)

def get_audio_duration(audio_path: Path) -> float:
    """
    Obtiene la duración del audio en segundos usando ffprobe.
    """
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        str(audio_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error obteniendo duración de {audio_path.name}: {e}")
        return 0.0

def optimize_audio(input_path: Path) -> Path:
    """
    Optimiza el audio usando ffmpeg directamente via subprocess.
    """
    logger.info(f"Optimizando audio: {input_path}")
    try:
        optimized_path = input_path.parent / f"optimized_{input_path.name}"
        # Asegurar extensión mp3
        optimized_path = optimized_path.with_suffix('.mp3')
        
        # Comando ffmpeg
        cmd = [
            "ffmpeg",
            "-y", # Sobreescribir
            "-i", str(input_path),
            "-ac", TARGET_CHANNELS,
            "-ar", TARGET_SAMPLE_RATE,
            "-b:a", TARGET_BITRATE,
            "-vn", # No video
            str(optimized_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error ffmpeg optimizando: {result.stderr}")
            raise RuntimeError("Fallo en optimización ffmpeg")
            
        original_size = input_path.stat().st_size / (1024 * 1024)
        new_size = optimized_path.stat().st_size / (1024 * 1024)
        logger.info(f"Optimización completada. Original: {original_size:.2f}MB, Optimizado: {new_size:.2f}MB")
        
        return optimized_path
    
    except Exception as e:
        logger.error(f"Error crítico en optimize_audio: {e}")
        raise

def split_audio_smart(audio_path: Path, segment_time: int = CHUNK_TIME_SECONDS) -> List[Path]:
    """
    Divide el audio en chunks usando ffmpeg segment.
    """
    logger.info(f"Procesando chunks ({segment_time}s) para: {audio_path}")
    
    output_pattern = audio_path.parent / f"chunk_%03d_{audio_path.name}"
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-c", "copy", # Copiar stream para ser rápido sin recodificar
        str(output_pattern)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
             logger.error(f"Error ffmpeg chunking: {result.stderr}")
             raise RuntimeError("Fallo en chunking ffmpeg")
             
        # Recopilar chunks generados
        chunk_prefix = f"chunk_"
        chunks = sorted([
            f for f in audio_path.parent.iterdir() 
            if f.name.startswith(chunk_prefix) and f.name.endswith(audio_path.name)
        ])
        
        logger.info(f"Se generaron {len(chunks)} chunks.")
        return chunks
        
    except Exception as e:
        logger.error(f"Error crítico en split_audio_smart: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def transcribe_chunk(client, chunk_path: Path, model: str, prompt_text: str = None) -> str:
    """
    Transcribe un chunk individual usando la API de OpenAI.
    """
    logger.info(f"Transcribiendo chunk: {chunk_path.name} con modelo {model}")
    try:
        # Verificar tamaño antes de enviar (<25MB)
        size_mb = chunk_path.stat().st_size / (1024 * 1024)
        if size_mb > 24:
            logger.warning(f"Chunk {chunk_path.name} es muy grande ({size_mb:.2f}MB). Podría fallar.")
            
        with open(chunk_path, "rb") as audio_file:
            # Prepare arguments
            kwargs = {
                "model": model,
                "file": audio_file
            }
            if prompt_text:
                kwargs["prompt"] = prompt_text
                
            response = client.audio.transcriptions.create(**kwargs)
            return response.text
    except Exception as e:
        logger.error(f"Error al transcribir chunk {chunk_path.name}: {e}")
        raise

def transcribe_with_gemini(audio_path: Path, model_name: str, prompt_text: str = None) -> str:
    """
    Transcribe un archivo completo usando Google Gemini.
    """
    logger.info(f"Transcribiendo con Gemini ({model_name}): {audio_path.name}")
    try:
        # Subir archivo
        audio_file = genai.upload_file(path=str(audio_path))
        
        # Configurar modelo
        model = genai.GenerativeModel(model_name)
        
        prompt = "Transcribe este audio íntegramente. Mantén los turnos de palabra si es posible y usa puntuación correcta."
        if prompt_text:
            prompt = f"{prompt} Contexto adicional: {prompt_text}"
            
        response = model.generate_content(
            [prompt, audio_file],
            request_options={"timeout": 7200}
        )
        return response.text
    except Exception as e:
        logger.error(f"Error en Gemini ({audio_path.name}): {e}")
        raise

def process_file(client, input_file: Path, output_dir: Path, model: str, move_to: Path = None, delete_original: bool = False) -> dict:
    """
    Proceso principal para un archivo.
    Devuelve un diccionario con las rutas del archivo de audio optimizado y la transcripción.
    """
    # === ANALISIS DE NOMBRE DE ARCHIVO ===
    # Formato esperado: "CODIGO [espacio] FECHAYYYYMMDD [espacio] IDIOMAS?.ext"
    # Ej: "R2601 20260109 español valenciano.mp3"
    
    filename_stem = input_file.stem # "R2601 20260109 español valenciano"
    
    # Regex para capturar: (Grupo 1: Todo antes de la fecha) (Grupo 2: La fecha 8 digitos) (Grupo 3: Todo después)
    match = re.search(r"^(.*?)\s*(\d{8})(.*)$", filename_stem)
    
    folder_name = ""
    clean_stem = filename_stem # Default por si falla el regex
    languages_prompt = ""
    
    if match:
        pre_date = match.group(1).strip(" []")
        date_part = match.group(2)
        post_date = match.group(3).strip() # Aqui estarian los idiomas
        
        # 1. Nombre de la carpeta (Pre-Date)
        folder_name = pre_date
        if not folder_name:
             folder_name = "OTROS"
             
        # 2. Nombre limpio de archivo (Pre-Date + Date)
        # Reconstruimos: "R2601 20260109"
        clean_stem = f"{pre_date} {date_part}".strip()
        
        # 3. Prompt de idiomas
        if post_date:
            languages_prompt = f"La transcripción debe ser precisa en estos idiomas: {post_date}."
            logger.info(f"Detectados idiomas/info extra: '{post_date}'. Usando como prompt.")
            tqdm.write(f"  -> Idiomas detectados: {post_date}")
            
    else:
        # Fallback si no hay fecha estándar
        folder_name = filename_stem.split(' ')[0].strip("[]")
        # Si no hay fecha, no limpiamos nada, usamos el original
        clean_stem = filename_stem

    # Actualizar directorio de salida apuntando a la subcarpeta
    output_dir = output_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=== Iniciando procesamiento de {input_file.name} ===")
    logger.info(f"Carpeta destino: {output_dir}")
    tqdm.write(f"Procesando: {input_file.name} -> Carpeta: {folder_name}/")

    
    temp_files = []
    final_audio_path = None
    output_file = None

    try:
        # 1. Optimizar
        tqdm.write("  -> Optimizando audio...", end="")
        # optimize_audio crea el archivo en el mismo directorio que el input
        temp_optimized = optimize_audio(input_file)
        
        # Comparar tamaños
        orig_size = input_file.stat().st_size
        opt_size = temp_optimized.stat().st_size
        
        if opt_size < (orig_size * 0.90):
            # Optimizada es significativamente más ligera (>10% ahorro), la usamos
            final_audio_name = f"{clean_stem}.mp3"
            final_audio_path = output_dir / final_audio_name
            
            if final_audio_path.exists():
                os.remove(final_audio_path)
                
            shutil.move(str(temp_optimized), str(final_audio_path))
            tqdm.write(f" Hecho (Optimizado: {opt_size/1024/1024:.2f}MB vs Original: {orig_size/1024/1024:.2f}MB -> Ahorro >10%, usamos MP3)")
        else:
            # Original es mejor o el ahorro es despreciable (<10%), mantenemos calidad original
            final_audio_name = f"{clean_stem}{input_file.suffix}"
            final_audio_path = output_dir / final_audio_name
            
            if final_audio_path.exists():
                os.remove(final_audio_path)
            
            # Copiamos para preservar original en su sitio (por si luego se mueve a Procesados)
            shutil.copy2(str(input_file), str(final_audio_path))
            
            # Borrar temp optimizado
            os.remove(temp_optimized)
            tqdm.write(f" Hecho (Original: {orig_size/1024/1024:.2f}MB vs Optimizado: {opt_size/1024/1024:.2f}MB -> Ahorro <10%, mantenemos Original)")
        
        # 2. Obtener duración y decidir troceado
        duration = get_audio_duration(final_audio_path)
        logger.info(f"Duración detectada: {duration:.2f}s")
        
        chunks = []
        is_gemini = model.startswith("gemini")
        
        if is_gemini:
            # Si Gemini y > 1h, troceamos en bloques de 40 min para asegurar estabilidad
            if duration > 3600:
                tqdm.write(f"  -> Audio largo detectado ({duration/3600:.1f}h). Dividiendo para Gemini...")
                sys.stdout.flush()
                chunks = split_audio_smart(final_audio_path, segment_time=2400) # 40 min
            else:
                # Audio corto, Gemini lo procesa de una pieza
                chunks = [final_audio_path]
        else:
            # OpenAI siempre usa chunks de 10 min
            chunks = split_audio_smart(final_audio_path, segment_time=CHUNK_TIME_SECONDS)
            
        if chunks:
            new_chunks = [c for c in chunks if c != final_audio_path]
            temp_files.extend(new_chunks)
        
        # 3. Transcribir
        full_transcript = []
        
        for i, chunk in enumerate(chunks):
            if is_gemini:
                if len(chunks) > 1:
                    tqdm.write(f"  -> Transcribiendo bloque {i+1}/{len(chunks)} con {model}...")
                    sys.stdout.flush()
                else:
                    tqdm.write(f"  -> Transcribiendo con {model}...")
                    sys.stdout.flush()
                    
                text = transcribe_with_gemini(chunk, model, prompt_text=languages_prompt)
                full_transcript.append(text)
            else:
                # OpenAI
                tqdm.write(f"  -> Transcribiendo {chunk.name}...")
                sys.stdout.flush()
                text = transcribe_chunk(client, chunk, model, prompt_text=languages_prompt)
                full_transcript.append(text)
        
        # 4. Guardar (Usando NOMBRE LIMPIO)
        # Separador para Gemini si hay varios trozos
        separator = "\n\n[Continuación...]\n\n" if (is_gemini and len(chunks) > 1) else "\n"
        final_text = separator.join(full_transcript)
        output_file = output_dir / f"{clean_stem}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_text)
            
        logger.info(f"Transcripción guardada en: {output_file}")
        logger.info(f"Audio optimizado guardado en: {final_audio_path}")
        # tqdm.write(f"  -> Guardado en {output_file.name}")

        return {
            "text_path": output_file,
            "audio_path": final_audio_path
        }
            
    except Exception as e:
        logger.error(f"Fallo crítico procesando {input_file.name}: {e}")
        tqdm.write(f"Error procesando {input_file.name}: Ver log.")
        raise e
    finally:
        # Limpieza de chunks (silenciosa)
        for f in temp_files:
            if f.exists() and f != final_audio_path and f != input_file: 
                try:
                    os.remove(f)
                except OSError:
                    pass

        # Mover archivo original si se solicitó y no hubo excepciones

        # Mover archivo original si se solicitó y no hubo excepciones
        if input_file.exists(): # Ensure input still exists
            if delete_original:
                 if output_file and output_file.exists(): # Safety check: output must exist
                     try:
                         os.remove(input_file)
                         logger.info(f"Archivo original eliminado: {input_file}")
                         tqdm.write(f"  -> Original eliminado del input.")
                     except Exception as del_e:
                         logger.error(f"Error eliminando original: {del_e}")
            elif move_to:
                try:
                    if output_file and output_file.exists():
                         move_to.mkdir(parents=True, exist_ok=True)
                         dest_path = move_to / input_file.name
                         shutil.move(str(input_file), str(dest_path))
                         logger.info(f"Archivo original movido a: {dest_path}")
                         tqdm.write(f"  -> Original movido a {dest_path}")
                except Exception as move_e:
                    logger.error(f"Error moviendo archivo original: {move_e}")
                    tqdm.write(f"Error moviendo original: {move_e}")

def main():
    parser = argparse.ArgumentParser(description="Batch Audio Transcription with OpenAI Models")
    parser.add_argument("--input", default="input", help="Input directory containing audio files (default: input)")
    parser.add_argument("--output", default="output", help="Output directory for text files (default: output)")
    parser.add_argument("--file", help="Path to a single audio file to process (overrides --input directory mode)")
    parser.add_argument("--model", default="gpt-4o-mini-transcribe", 
                        choices=["whisper-1", "gpt-4o-mini-transcribe", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"], 
                        help="Model to use (default: gpt-4o-mini-transcribe)")
    parser.add_argument("--move-processed-to", help="Directory to move original files after successful processing")
    parser.add_argument("--delete-original", action="store_true", help="Delete original file after successful processing")
    
    args = parser.parse_args()
    
    # Check dependencies
    check_ffmpeg()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dir = Path(args.move_processed_to) if args.move_processed_to else None
    
    # Inicializar cliente según el modelo
    openai_client = None
    
    if args.model.startswith("gemini"):
        google_key = os.environ.get('GOOGLE_API_KEY')
        if not google_key:
            print("Error: No se encontró GOOGLE_API_KEY en las variables de entorno.")
            sys.exit(1)
        genai.configure(api_key=google_key)
    else:
        openai_client = get_openai_client()

    # MODO 1: Archivo único
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: El archivo {file_path} no existe.")
            sys.exit(1)
        
        print(f"Procesando archivo único: {file_path.name} usando modelo {args.model}")
        process_file(openai_client, file_path, output_dir, args.model, move_to=processed_dir, delete_original=args.delete_original)
        print("\nProcesamiento finalizado.")
        return

    # MODO 2: Directorio (Batch)
    input_dir = Path(args.input)
    
    if not input_dir.exists():
        print(f"Error: El directorio {input_dir} no existe.")
        sys.exit(1)
        
    supported_extensions = {'.mp3', '.wav', '.m4a', '.mp4', '.mpeg', '.mpga', '.webm', '.mov', '.avi', '.mkv', '.flv', '.wmv'}
    
    files_to_process = [f for f in input_dir.iterdir() if f.suffix.lower() in supported_extensions]
    
    if not files_to_process:
        print(f"No se encontraron archivos válidos en {input_dir}")
        return

    print(f"Encontrados {len(files_to_process)} archivos. Iniciando con modelo {args.model}...")
    
    # Barra de progreso total
    for audio_file in tqdm(files_to_process, desc="Progreso Total", unit="archivos"):
        process_file(openai_client, audio_file, output_dir, args.model, move_to=processed_dir, delete_original=args.delete_original)
    
    print("\nProcesamiento finalizado.")

if __name__ == "__main__":
    main()
