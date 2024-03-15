# !pip install pytesseract 

import logging
from pathlib import Path
from PIL import Image
import pytesseract

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Paths and directory names
current_directory = Path.cwd()
photos_dir_name = 'data/photos'
transcripts_dir_name = 'transcripts'
photos_dir_path = current_directory / photos_dir_name
transcripts_dir_path = current_directory / transcripts_dir_name

# Ensure the transcripts directory exists
transcripts_dir_path.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_next_file_id(directory_path: Path):
    existing_files = [f.stem for f in directory_path.glob('text_extract_*.txt')]
    existing_ids = [int(f.split('_')[-1]) for f in existing_files if f.split('_')[-1].isdigit()]
    next_id = max(existing_ids) + 1 if existing_ids else 1
    return next_id

def get_filenames(directory_path: Path):
    if directory_path.exists() and directory_path.is_dir():
        return [entry for entry in directory_path.iterdir() if entry.is_file()]
    else:
        logging.error(f"Directory '{directory_path}' does not exist.")
        return []

def main():
    sorted_photos_filenames = get_filenames(photos_dir_path)
    logging.info("Files to be transcribed:")
    for path in sorted_photos_filenames:
        logging.info(path)

    extracted_texts = []
    for path in sorted_photos_filenames:
        try:
            img = Image.open(path)
            text = pytesseract.image_to_string(img)
            extracted_texts.append(f"File: {path.name}\n{text}")
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")

    if extracted_texts:
        next_id = get_next_file_id(transcripts_dir_path)
        transcripts_file_name = f"text_extract_{next_id}.txt"
        transcripts_file_path = transcripts_dir_path / transcripts_file_name
        transcripts_file_path.write_text("\n\n---\n\n".join(extracted_texts))
        logging.info(f"Transcribed text has been saved to {transcripts_file_path}")

if __name__ == "__main__":
    main()