# !pip install moviepy SpeechRecognition pydub m3u8

import logging
from pathlib import Path
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import concurrent.futures
import shutil
import m3u8
import requests
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Paths and directory names
current_directory = Path.cwd()

print(current_directory)

videos_dir_name = 'data/videos'
transcripts_dir_name = 'transcripts'
chunks_dir_name = 'data/audio_chunks'
processed_videos_file_name = 'data/processed_videos.txt'

# Directories
videos_dir_path = current_directory / videos_dir_name
transcripts_dir_path = current_directory / transcripts_dir_name
chunks_dir_path = current_directory / chunks_dir_name
processed_videos_path = current_directory / processed_videos_file_name

# Ensure directories exist
transcripts_dir_path.mkdir(exist_ok=True)
chunks_dir_path.mkdir(exist_ok=True)

# Create or append to the processed videos file
processed_videos_path.touch()

def get_processed_videos():
    with processed_videos_path.open('r') as file:
        return set(file.read().splitlines())

def mark_video_as_processed(video_name):
    with processed_videos_path.open('a') as file:
        file.write(f"{video_name}\n")

def get_next_file_id(directory_path: Path, prefix='audio_extract_'):
    existing_files = [f.stem for f in directory_path.glob(f'{prefix}*.txt')]
    existing_ids = [int(f.replace(prefix, '')) for f in existing_files if f.replace(prefix, '').isdigit()]
    next_id = max(existing_ids) + 1 if existing_ids else 1
    return next_id

def get_filenames(directory_path: Path, processed_videos):
    if directory_path.exists() and directory_path.is_dir():
        return [entry for entry in directory_path.iterdir() if entry.is_file() and entry.suffix in ['.mp4', '.m3u8'] and entry.name not in processed_videos]
    logging.error(f"Directory '{directory_path}' does not exist.")
    return []

def convert_m3u8_to_mp4(video_file):
    output_file = video_file.with_suffix('.mp4')
    ffmpeg_path = 'C:/ffmpeg/bin/ffmpeg.exe'  # Change this to your ffmpeg path
    command = [
        ffmpeg_path,
        '-protocol_whitelist', 'file,http,https,tcp,tls',
        '-i', str(video_file),
        '-c', 'copy',
        str(output_file)
    ]
    try:
        subprocess.run(command, check=True)
        return output_file
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to convert {video_file.name} to mp4: {e}")
        return None

def segment_audio(video_file):
    if video_file.suffix == '.m3u8':
        converted_file = convert_m3u8_to_mp4(video_file)
        if not converted_file:
            logging.error(f"Failed to convert m3u8 file {video_file.name}")
            return [], None
        video_file = converted_file
    
    video_clip = VideoFileClip(str(video_file))
    audio_file = chunks_dir_path / (video_file.stem + '.wav')
    video_clip.audio.write_audiofile(str(audio_file), logger=None)
    
    sound = AudioSegment.from_file(str(audio_file))
    chunks = split_on_silence(sound, min_silence_len=1000, silence_thresh=sound.dBFS-14, keep_silence=500)
    chunk_files = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_file = chunks_dir_path / f"{audio_file.stem}_chunk{i}.wav"
        chunk.export(chunk_file, format="wav")
        chunk_files.append(chunk_file)
    
    return chunk_files, audio_file.stem

def transcribe_audio_chunk(chunk_file: Path) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(str(chunk_file)) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            logging.warning(f"Google Speech Recognition could not understand audio {chunk_file.name}.")
        except sr.RequestError as e:
            logging.error(f"Could not request results from Google Speech Recognition service; {e}.")
    return ""

def clean_up_chunks(chunk_files, video_file):
    for chunk_file in chunk_files:
        chunk_file.unlink()

    audio_file = chunks_dir_path / (video_file.stem + '.wav')
    audio_file.unlink()

def transcribe_video(video_file: Path, processed_videos):
    if video_file.name in processed_videos:
        logging.info(f"Skipping {video_file.name}, already processed.")
        return
    
    logging.info(f"Processing video: {video_file.name}")
    chunk_files, stem = segment_audio(video_file)
    if not chunk_files:
        logging.error(f"Failed to segment audio for {video_file.name}")
        return
    
    all_text = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(transcribe_audio_chunk, chunk_file): chunk_file for chunk_file in chunk_files}
        for future in concurrent.futures.as_completed(futures):
            chunk_file = futures[future]
            try:
                text = future.result()
                if text:
                    all_text.append(text)
            except Exception as exc:
                logging.error(f"{chunk_file.name} generated an exception: {exc}")
    
    clean_up_chunks(chunk_files, video_file)
    
    if all_text:
        extracted_text = "\n".join(all_text)
        next_id = get_next_file_id(transcripts_dir_path)
        output_file_name = f"audio_extract_{next_id}.txt"
        output_file = transcripts_dir_path / output_file_name
        output_file.write_text(extracted_text)
        mark_video_as_processed(video_file.name)
        logging.info(f"Transcribed text has been saved to {output_file}")

def main():
    processed_videos = get_processed_videos()
    video_files = get_filenames(videos_dir_path, processed_videos)
    for video_file in video_files:
        transcribe_video(video_file, processed_videos)

if __name__ == "__main__":
    main()
