# !pip install moviepy SpeechRecognition pydub

import logging
from pathlib import Path
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import concurrent.futures
import shutil

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
    """Read the list of already processed videos."""
    with processed_videos_path.open('r') as file:
        return set(file.read().splitlines())

def mark_video_as_processed(video_name):
    """Mark a video as processed by adding it to the list."""
    with processed_videos_path.open('a') as file:
        file.write(f"{video_name}\n")

def get_next_file_id(directory_path: Path, prefix='audio_extract_'):
    """Get the next file ID for naming output files."""
    existing_files = [f.stem for f in directory_path.glob(f'{prefix}*.txt')]
    existing_ids = [int(f.replace(prefix, '')) for f in existing_files if f.replace(prefix, '').isdigit()]
    next_id = max(existing_ids) + 1 if existing_ids else 1
    return next_id

def get_filenames(directory_path: Path, processed_videos):
    """Filter out already processed videos."""
    if directory_path.exists() and directory_path.is_dir():
        return [entry for entry in directory_path.iterdir() if entry.is_file() and entry.suffix == '.mp4' and entry.name not in processed_videos]
    logging.error(f"Directory '{directory_path}' does not exist.")
    return []

def segment_audio(video_file):
    video_clip = VideoFileClip(str(video_file))
    audio_file = chunks_dir_path / (video_file.stem + '.wav')
    video_clip.audio.write_audiofile(str(audio_file), logger=None)  # Disable moviepy logging
    
    sound = AudioSegment.from_file(str(audio_file))
    chunks = split_on_silence(sound, min_silence_len=1000, silence_thresh=sound.dBFS-14, keep_silence=500)
    chunk_files = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_file = chunks_dir_path / f"{audio_file.stem}_chunk{i}.wav"
        chunk.export(chunk_file, format="wav")
        chunk_files.append(chunk_file)
    
    # Make sure to return exactly two items: the list of chunk files and the audio file stem
    return chunk_files, audio_file.stem

def transcribe_audio_chunk(chunk_file: Path) -> str:
    """Transcribe a single chunk of audio to text."""
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

def clean_up_chunks(chunk_files):
    """Remove chunk files after processing."""
    for chunk_file in chunk_files:
        chunk_file.unlink()

def transcribe_video(video_file: Path, processed_videos):
    """Include a check to skip already processed videos."""
    if video_file.name in processed_videos:
        logging.info(f"Skipping {video_file.name}, already processed.")
        return
    
    logging.info(f"Processing video: {video_file.name}")
    chunk_files, stem = segment_audio(video_file)
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
    
    clean_up_chunks(chunk_files)  # Clean up after transcription
    
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