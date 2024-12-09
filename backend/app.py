import os
import json
import base64
import tempfile
import re
import logging
from flask import Flask, request, jsonify, send_from_directory
from google.cloud import texttospeech
from moviepy.editor import AudioFileClip, concatenate_audioclips
from flask_cors import CORS
from io import StringIO

app = Flask(__name__)

MAX_CHARS = 5000
OUTPUT_DIR = "./output"
RETRY_LIMIT = 3  # Number of retries for failed chunks

# Enable CORS
CORS(app)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@app.route("/process-text", methods=["POST"])
def process_text():
    logs = []
    try:
        result, logs = _process_text_internal()
        # Get the full URL for the file
        absolute_url = request.host_url.strip("/") + result.json["audioUrl"]
        return jsonify({"audioUrl": absolute_url, "logs": logs}), 200
    except Exception as e:
        logging.error("Error occurred. Sending logs to the UI.")
        error_message = str(e).split("Logs:")[0].strip()  # Extract error message before logs
        logs = str(e).split("Logs:")[1].strip() if "Logs:" in str(e) else logs
        return jsonify({"message": error_message, "logs": logs}), 500

def capture_logs(func):
    """Decorator to capture logs for a function."""
    def wrapper(*args, **kwargs):
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            result = func(*args, **kwargs)
            logs = log_stream.getvalue()
            logger.removeHandler(handler)
            return result, logs
        except Exception as e:
            logs = log_stream.getvalue()
            logger.removeHandler(handler)
            raise Exception(f"{e}\nLogs:\n{logs}")
    return wrapper

@capture_logs
def _process_text_internal():
    """Main processing logic wrapped with log capture."""
    # Extract input text from the request
    data = request.get_json()
    text_input = data.get("text", None)

    if not text_input:
        raise ValueError("No valid text input provided.")

    logging.info(f"Text input received: {text_input[:100]}...")

    # Load Google Cloud credentials from the key.json file
    try:
        credentials_path = "./key.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        tts_client = texttospeech.TextToSpeechClient()
        logging.info("Google TTS client initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing Google TTS client: {e}")
        raise

    # Split text into chunks for processing
    try:
        parsed_turns = parse_dialogue(text_input)
        logging.debug(f"Parsed turns: {parsed_turns}")
        if not parsed_turns:
            raise ValueError("No valid turns were parsed from the input text.")

        chunks = split_into_chunks(parsed_turns)
        logging.debug(f"Generated chunks: {chunks}")
        if not chunks:
            raise ValueError("No valid chunks were created from the parsed turns.")
    except Exception as e:
        logging.error(f"Error parsing or chunking input: {e}")
        raise ValueError("Error processing input text.")

    logging.info(f"Split input into {len(chunks)} chunks.")

    # Synthesize audio for each chunk
    temp_audio_files = []
    for i, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {i + 1}/{len(chunks)}")
        try:
            audio_file = synthesize_chunk_with_retry(tts_client, chunk, i)
            temp_audio_files.append(audio_file)
        except Exception as e:
            logging.error(f"Error synthesizing chunk {i + 1}: {e}")
            raise ValueError(f"Error processing chunk {i + 1}: {e}")

    # Concatenate audio files
    try:
        if not temp_audio_files:
            raise ValueError("No audio files were generated for concatenation.")
        concatenated_audio_file = concatenate_audio(temp_audio_files)
        logging.info("Audio concatenation successful.")
    except Exception as e:
        logging.error(f"Error concatenating audio files: {e}")
        raise ValueError("Error concatenating audio files.")

    # Move the final file to the output directory
    output_file_path = os.path.join(OUTPUT_DIR, "final_output.mp3")
    os.rename(concatenated_audio_file, output_file_path)

    # Cleanup temporary files
    cleanup_temp_files(temp_audio_files)

    # Return the URL to download the file
    return jsonify({"audioUrl": f"/download/final_output.mp3"})


@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    """Serve the file from the output directory."""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


def synthesize_chunk_with_retry(tts_client, chunk, chunk_index):
    """
    Retry mechanism for synthesizing chunks that fail.
    """
    attempt = 0
    while attempt < RETRY_LIMIT:
        try:
            logging.info(f"Attempt {attempt + 1} for chunk {chunk_index + 1}")
            return synthesize_chunk(tts_client, chunk, chunk_index)
        except Exception as e:
            logging.error(f"Retry {attempt + 1} for chunk {chunk_index + 1} failed: {e}")
            if attempt < RETRY_LIMIT - 1:
                chunk = retry_chunk(chunk)
            attempt += 1
    raise ValueError(f"Failed to synthesize chunk {chunk_index + 1} after {RETRY_LIMIT} attempts.")


def retry_chunk(chunk):
    """
    Modify the chunk by progressively reducing complexity.
    If still too long, split into individual turns.
    """
    # Step 1: Remove punctuations
    for turn in chunk:
        turn["text"] = re.sub(r"[^\w\s]", "", turn["text"])
    logging.info(f"Retrying chunk with reduced punctuations: {chunk}")

    # Step 2: If still too large, split into individual turns
    if len(chunk) > 1:
        logging.warning("Splitting chunk into individual turns due to repeated failures.")
        return [[turn] for turn in chunk]

    return chunk

def log_chunk_details(chunks):
    for i, chunk in enumerate(chunks):
        chunk_size = sum(len(turn["text"]) + len(turn["speaker"]) + 2 for turn in chunk)
        logging.info(f"Chunk {i + 1}: {len(chunk)} turns, {chunk_size} bytes")


def sanitize_text(text):
    """Sanitize text to remove unsupported characters."""
    sanitized = text.replace("\n", " ").replace("\r", "").strip()
    # Remove unsupported characters (e.g., emojis or control characters)
    sanitized = re.sub(r"[^\x00-\x7F]+", "", sanitized)
    return sanitized


def synthesize_chunk(tts_client, chunk, chunk_index):
    """
    Synthesizes a chunk of text using MultiSpeakerMarkup and saves it to a temporary file.
    """
    logging.info(f"Processing chunk {chunk_index + 1}...")

    # Construct the MultiSpeakerMarkup for the chunk
    try:
        multi_speaker_markup = texttospeech.MultiSpeakerMarkup(
            turns=[
                texttospeech.MultiSpeakerMarkup.Turn(
                    text=turn["text"], speaker=turn["speaker"]
                )
                for turn in chunk
            ]
        )

        # Set the input for synthesis
        synthesis_input = texttospeech.SynthesisInput(
            multi_speaker_markup=multi_speaker_markup
        )

        # Configure the voice parameters
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name="en-US-Studio-MultiSpeaker"
        )

        # Configure the audio output
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        # Call the TTS API
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        logging.info(f"Synthesize speech successful for chunk {chunk_index + 1}")
    except Exception as e:
        logging.error(f"Error in TTS API for chunk {chunk_index + 1}: {e}")
        raise

    # Save the audio content to a temporary file
    temp_file = f"./chunk_{chunk_index}.mp3"
    with open(temp_file, "wb") as out:
        out.write(response.audio_content)
    logging.info(f"Audio chunk {chunk_index + 1} written to {temp_file}")
    return temp_file


def concatenate_audio(audio_files):
    clips = [AudioFileClip(file) for file in audio_files]
    concatenated_clip = concatenate_audioclips(clips)
    output_file = "./temp_final_output.mp3"
    concatenated_clip.write_audiofile(output_file, codec="libmp3lame")
    return output_file

def parse_dialogue(text):
    """
    Parses the input text into a structured format compatible with MultiSpeakerMarkup.
    The input is expected to follow the format: 'Speaker N: text'.
    """
    turns = []
    for line in text.strip().split("\n"):
        match = re.match(r"(Speaker \d+):\s*(.*)", line, re.IGNORECASE)
        if match:
            speaker_label, utterance = match.groups()
            speaker_code = map_speaker_label_to_code(speaker_label)
            if speaker_code and utterance:
                turns.append({"speaker": speaker_code, "text": utterance})
        else:
            logging.warning(f"Unrecognized line format: {line}")
    if not turns:
        raise ValueError("No valid dialogue structure found in input text.")
    logging.debug(f"Parsed turns: {turns}")
    return turns


def split_into_chunks(parsed_turns):
    """
    Splits parsed dialogue into chunks of less than MAX_CHARS.
    Ensures chunks are generated properly even for large single turns.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for turn in parsed_turns:
        turn_length = len(turn["text"]) + len(turn["speaker"]) + 2  # Include ": "

        # Handle cases where a single turn exceeds MAX_CHARS
        if turn_length > MAX_CHARS:
            logging.warning(f"Turn exceeds MAX_CHARS: {turn}")
            # Split the turn into smaller chunks
            split_text = split_large_turn(turn["text"], MAX_CHARS - len(turn["speaker"]) - 2)
            for part in split_text:
                chunks.append([{"speaker": turn["speaker"], "text": part}])
            continue

        # Add turn to the current chunk if it fits
        if current_length + turn_length <= MAX_CHARS:
            current_chunk.append(turn)
            current_length += turn_length
        else:
            # Save the current chunk and start a new one
            chunks.append(current_chunk)
            current_chunk = [turn]
            current_length = turn_length

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Additional Split: Ensure chunks don't exceed TTS API byte limit
    validated_chunks = []
    for chunk in chunks:
        validated_chunks.extend(split_large_chunk_by_bytes(chunk))
    return validated_chunks

def split_large_chunk_by_bytes(chunk, max_turns=3):
    """
    Ensures a chunk's total size does not exceed 5000 bytes and limits the number of turns.
    Splits into smaller sub-chunks if needed.
    """
    sub_chunks = []
    current_sub_chunk = []
    current_sub_size = 0

    for turn in chunk:
        turn_size = len(turn["text"]) + len(turn["speaker"]) + 2  # Text + speaker + ": "
        
        # Add turn if it fits within size and turn count limits
        if (current_sub_size + turn_size <= 5000) and (len(current_sub_chunk) < max_turns):
            current_sub_chunk.append(turn)
            current_sub_size += turn_size
        else:
            # Save the current sub-chunk and start a new one
            sub_chunks.append(current_sub_chunk)
            current_sub_chunk = [turn]
            current_sub_size = turn_size

    # Add the last sub-chunk
    if current_sub_chunk:
        sub_chunks.append(current_sub_chunk)

    return sub_chunks



def split_large_turn(text, max_length):
    """
    Splits a single large turn into smaller chunks that fit within max_length.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_length:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def map_speaker_label_to_code(speaker_label):
    """
    Maps speaker labels to corresponding codes for Google TTS.
    """
    speaker_map = {"Speaker 1": "R", "Speaker 2": "S"}
    return speaker_map.get(speaker_label, None)


def cleanup_temp_files(files):
    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            logging.error(f"Error deleting temp file {file}: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
