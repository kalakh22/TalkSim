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

MAX_CHARS = 2000  # Maximum characters per chunk
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
    Retry mechanism for chunks that fail due to size or complexity. Splits long turns into smaller sub-turns.
    """
    refined_chunk = []
    for turn in chunk:
        if len(turn["text"]) + len(turn["speaker"]) + 2 > MAX_CHARS:
            # Split the long turn into smaller sub-turns
            sub_turns = split_large_turn(turn["text"], MAX_CHARS - len(turn["speaker"]) - 2)
            refined_chunk.extend([{"speaker": turn["speaker"], "text": sub_turn} for sub_turn in sub_turns])
        else:
            refined_chunk.append(turn)

    return refined_chunk



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

    # Validate chunk structure
    if not all("speaker" in turn and "text" in turn for turn in chunk):
        raise ValueError(f"Invalid chunk structure: {chunk}")

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
    Lines without 'Speaker X:' will be appended to the previous speaker's text.
    """
    turns = []
    current_speaker = None
    current_text = []

    for line in text.strip().split("\n"):
        match = re.match(r"(Speaker \d+):\s*(.*)", line, re.IGNORECASE)
        if match:
            # Save the previous speaker's turn if present
            if current_speaker and current_text:
                turns.append({"speaker": map_speaker_label_to_code(current_speaker), "text": " ".join(current_text)})
                current_text = []

            # Start a new speaker's turn
            current_speaker, text = match.groups()
            current_text.append(text)
        else:
            # Append text to the current speaker
            if current_speaker:
                current_text.append(line.strip())

    # Add the final speaker's turn
    if current_speaker and current_text:
        turns.append({"speaker": map_speaker_label_to_code(current_speaker), "text": " ".join(current_text)})

    if not turns:
        raise ValueError("No valid dialogue structure found in input text.")

    logging.debug(f"Parsed turns: {turns}")
    return turns




def split_into_chunks(parsed_turns):
    """
    Splits parsed dialogue into chunks, ensuring no single turn exceeds the character limit
    and that each chunk fits within the max character limit.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for turn in parsed_turns:
        turn_length = len(turn["text"]) + len(turn["speaker"]) + 2  # Include ": "

        # If the turn exceeds the max length, split it into smaller sub-turns
        if turn_length > MAX_CHARS:
            sub_turns = split_large_turn(turn["text"], MAX_CHARS - len(turn["speaker"]) - 2)
            for sub_turn in sub_turns:
                new_turn = {"speaker": turn["speaker"], "text": sub_turn}
                if current_length + len(sub_turn) + len(turn["speaker"]) + 2 <= MAX_CHARS:
                    current_chunk.append(new_turn)
                    current_length += len(sub_turn) + len(turn["speaker"]) + 2
                else:
                    chunks.append(current_chunk)
                    current_chunk = [new_turn]
                    current_length = len(sub_turn) + len(turn["speaker"]) + 2
            continue

        # Add the turn to the current chunk if it fits
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

    return chunks





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
    Splits a single long turn into smaller sub-turns that fit within the max_length limit.
    """
    words = text.split()
    sub_turns = []
    current_sub_turn = []

    for word in words:
        # Add word to current sub-turn if it fits
        if len(" ".join(current_sub_turn + [word])) <= max_length:
            current_sub_turn.append(word)
        else:
            # Save the current sub-turn and start a new one
            sub_turns.append(" ".join(current_sub_turn))
            current_sub_turn = [word]

    # Add the last sub-turn
    if current_sub_turn:
        sub_turns.append(" ".join(current_sub_turn))

    return sub_turns




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
