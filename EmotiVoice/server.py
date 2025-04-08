import os
import subprocess
import nltk
from flask import Flask, request, jsonify, send_file, after_this_request # Added after_this_request
from pydub import AudioSegment
from transformers import pipeline
import logging
from datetime import datetime
import re
import uuid
from werkzeug.utils import secure_filename
from flask_cors import CORS
import base64 # <--- ADDED THIS IMPORT
import shutil # <--- ADDED THIS IMPORT
import glob   # <--- ADDED THIS IMPORT
import json
# use cors to allow cross-origin requests



# Text Extraction Libraries
try:
    import pypdf
except ImportError:
    logging.error("pypdf library not found. Install using: pip install pypdf")
    pypdf = None
try:
    from docx import Document
except ImportError:
    logging.error("python-docx library not found. Install using: pip install python-docx")
    Document = None

# --- Basic Setup ---
app = Flask(__name__) # Define app ONCE here
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- Configuration ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_DIR, 'uploads')
OUTPUT_DIR = os.path.join(APP_DIR, "output") # General output, might not be used directly by endpoints
SAVED_BOOKS_DIR = os.path.join(APP_DIR, "saved_books") 
# SCRIPTS_DIR = os.path.join(APP_DIR, "scripts") # Not explicitly used if script is in APP_DIR
AUDIO_DIR = os.path.join(APP_DIR, "audio_files") # Not explicitly used by these endpoints
TEMP_PROCESSING_DIR = os.path.join(APP_DIR, "temp_batch_processing") # For batch TTS

# --- Script Configuration ---
# Use the specified script name for BOTH single and batch (assuming it handles both or you have one script)
TTS_SCRIPT_NAME = "script.sh"
TTS_SCRIPT_PATH = os.path.join(APP_DIR, TTS_SCRIPT_NAME) # Path to the script.sh

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(SCRIPTS_DIR, exist_ok=True) # Not needed if script is in APP_DIR
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True) # Create base temp dir
os.makedirs(SAVED_BOOKS_DIR, exist_ok=True) # Create saved books dir

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}



CORS(app, resources={
    # This line SHOULD cover /api/save-book, /api/list-books, etc.
    r"/api/*": {"origins": "http://localhost:3000"},
    r"/synthesize_speech": {"origins": "http://localhost:3000"},
})

# --- Emotion Analysis Setup ---
try:
    logging.info("Loading emotion classification model...")
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        # Consider adding truncation=True, max_length=512 if model has limits
    )
    logging.info("Emotion classification model loaded.")
except Exception as e:
    logging.error(f"Failed to load emotion model: {e}")
    emotion_classifier = None

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logging.info("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    logging.info("NLTK 'punkt' downloaded.")

# --- Helper Functions ---

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    if not pypdf: raise RuntimeError("pypdf library not installed.")
    text = ""
    try:
        reader = pypdf.PdfReader(filepath)
        if reader.is_encrypted:
            logging.warning(f"PDF file is encrypted: {filepath}. Cannot extract text.")
            return "" # Cannot process encrypted PDFs without password

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                 text += page_text + "\n" # Add newline between pages
    except pypdf.errors.PdfReadError as pdf_err:
        logging.error(f"Error reading PDF structure {filepath}: {pdf_err}")
        raise ValueError(f"Could not read PDF file structure: {pdf_err}") from pdf_err
    except Exception as e:
        logging.error(f"Error reading PDF {filepath}: {e}", exc_info=True)
        raise ValueError(f"Could not process PDF file: {e}") from e
    return text

def extract_text_from_docx(filepath):
    """Extracts text from a DOCX file."""
    if not Document: raise RuntimeError("python-docx library not installed.")
    text = ""
    try:
        doc = Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logging.error(f"Error reading DOCX {filepath}: {e}", exc_info=True)
        raise ValueError(f"Could not process DOCX file: {e}") from e
    return text

def extract_text_from_txt(filepath):
    """Extracts text from a TXT file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e_utf8:
        logging.warning(f"Could not read TXT {filepath} as UTF-8: {e_utf8}. Trying latin-1.")
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e_latin1:
            logging.error(f"Error reading TXT {filepath} with UTF-8 and latin-1: {e_latin1}", exc_info=True)
            raise ValueError(f"Could not read TXT file with common encodings: {e_latin1}") from e_latin1

def get_emotion_scores(text):
    """Gets emotion scores using the loaded pipeline."""
    if not emotion_classifier:
        raise RuntimeError("Emotion classifier model not loaded.")

    # Some models might have token limits, handle potential errors
    try:
        results = emotion_classifier(text)
        # The pipeline might return [[{'label':.., 'score':..}]] or [{'label':.., 'score':..}]
        if results and isinstance(results, list) and len(results) > 0:
             inner_result = results[0]
             if isinstance(inner_result, list): # Handles nested list case
                 return inner_result
             elif isinstance(inner_result, dict): # Handles case where it's just one dict (less likely for return_all_scores)
                 return results # Return the list containing the single dict
             else:
                 logging.warning(f"Emotion classifier returned unexpected inner structure: {type(inner_result)} for text: {text[:50]}...")
                 return []
        else:
            logging.warning(f"Emotion classifier returned empty or unexpected result for text: {text[:50]}...")
            return []
    except Exception as e:
        logging.error(f"Error during emotion classification pipeline for text '{text[:50]}...': {e}", exc_info=False)
        return [] # Return empty list on pipeline error

def get_dominant_emotion(scores_list, target_emotions):
    """Finds the emotion with the highest score among the target emotions."""
    best_emotion = 'neutral' # Default
    max_score = -1.0

    if not isinstance(scores_list, list):
         logging.warning(f"Invalid input to get_dominant_emotion: expected list, got {type(scores_list)}")
         return best_emotion # Return default if input is not a list

    for score_dict in scores_list:
         # Check if score_dict is a dictionary with the expected keys and types
         if (isinstance(score_dict, dict) and
             'label' in score_dict and isinstance(score_dict['label'], str) and
             'score' in score_dict and isinstance(score_dict['score'], (float, int))):
            label = score_dict['label']
            score = score_dict['score']
            if label in target_emotions and score > max_score:
                max_score = score
                best_emotion = label
         else:
             logging.warning(f"Skipping invalid score entry in get_dominant_emotion: {score_dict}")

    return best_emotion

def sort_key_for_output_files(filename):
    """Extracts the numerical index from filenames like 'output_1_....wav' or '1.wav'."""
    basename = os.path.basename(filename)
    # Try format: output_1_....wav
    match_prefix = re.search(r'output_(\d+)_.*\.wav$', basename)
    if match_prefix:
        return int(match_prefix.group(1))
    # Try format: 1.wav (assuming script generates simple numbered files)
    match_simple = re.search(r'^(\d+)\.wav$', basename)
    if match_simple:
        return int(match_simple.group(1))
    # Fallback for unexpected names, putting them at the end
    logging.warning(f"Could not extract sorting index from filename: {basename}")
    return float('inf')

# --- Flask Routes ---

@app.route('/')
def home():
     return jsonify({
        "message": "Welcome to the Manuscript Emotion & Speech API!",
        "endpoints": {
            "/api/process-manuscript": "POST: Processes uploaded manuscript (pdf, docx, txt) for text and emotion. Requires 'manuscript' file upload.",
            "/synthesize_speech": "POST: Generates audio for batch text/emotion/speaker inputs using script.sh. Expects JSON body.",
            # Removed old GET endpoints if they are replaced by POST /synthesize_speech
        }
    })

@app.route('/api/process-manuscript', methods=['POST'])
def process_manuscript():
    """
    Receives uploaded manuscript, extracts text, classifies emotion per sentence.
    """
    logging.info("Received POST request for /api/process-manuscript")

    if 'manuscript' not in request.files:
        logging.warning("No 'manuscript' file part found in the request.")
        return jsonify({"error": "No manuscript file part in the request"}), 400

    file = request.files['manuscript']

    if file.filename == '':
        logging.warning("No file selected (filename is empty).")
        return jsonify({"error": "No selected file"}), 400

    # Use a temporary file path within UPLOAD_FOLDER
    temp_filepath = None # Initialize to None

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        extension = original_filename.rsplit('.', 1)[1].lower()
        # Create a unique temporary filename to avoid collisions
        temp_filename = f"{uuid.uuid4()}.{extension}"
        temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)

        try:
            logging.info(f"Saving temporary file: {temp_filepath} (from: {original_filename})")
            file.save(temp_filepath)

            # --- Extract Text ---
            full_text = ""
            logging.info(f"Extracting text from .{extension} file...")
            if extension == 'pdf':
                full_text = extract_text_from_pdf(temp_filepath)
            elif extension == 'docx':
                full_text = extract_text_from_docx(temp_filepath)
            elif extension == 'txt':
                 full_text = extract_text_from_txt(temp_filepath)
            # No 'else' needed due to allowed_file check

            # Check if text extraction yielded anything meaningful
            if not full_text or full_text.isspace():
                 logging.warning(f"No text extracted or only whitespace found in file: {original_filename}")
                 # No need to clean up here, finally block will handle it
                 return jsonify({"error": "Could not extract text from the file or the file appears to be empty."}), 400

            logging.info(f"Extracted text length: {len(full_text)}. Tokenizing sentences...")

            # --- Tokenize into Sentences ---
            normalized_text = re.sub(r'[\r\n]+', ' ', full_text) # Replace newlines with spaces
            sentences = nltk.sent_tokenize(normalized_text)
            logging.info(f"Tokenized into {len(sentences)} potential sentences. Classifying emotions...")

            # --- Classify Emotions ---
            processed_lines = []
            target_emotions = {'joy', 'anger', 'sadness', 'surprise', 'neutral', 'fear', 'disgust'} # Define target set

            if not emotion_classifier:
                 logging.error("Emotion classifier not loaded, cannot process manuscript.")
                 # No need to clean up here, finally block will handle it
                 return jsonify({"error": "Emotion analysis model is not available on the server."}), 503 # Service Unavailable

            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                # Skip empty or punctuation-only strings resulting from tokenization/stripping
                if not sentence or all(not c.isalnum() for c in sentence):
                    continue

                try:
                    # Truncate very long sentences *only* for classification to avoid model errors
                    max_classify_len = 512 # Adjust based on model limits if known
                    if len(sentence) > max_classify_len:
                        logging.warning(f"Sentence {i+1} too long ({len(sentence)} chars), truncating for classification: '{sentence[:50]}...'")
                        sentence_to_classify = sentence[:max_classify_len]
                    else:
                        sentence_to_classify = sentence

                    scores_list = get_emotion_scores(sentence_to_classify)
                    dominant_emotion = get_dominant_emotion(scores_list, target_emotions)

                    processed_lines.append({
                        "id": str(uuid.uuid4()), # Add a unique ID for each line/sentence
                        "text": sentence,        # Return the original, untruncated sentence
                        "emotion": dominant_emotion
                    })

                except Exception as e_classify:
                    # Log error for this specific sentence but continue with others
                    logging.error(f"Error classifying sentence {i+1} ('{sentence[:50]}...'): {e_classify}", exc_info=False)
                    processed_lines.append({
                        "id": str(uuid.uuid4()),
                        "text": sentence,
                        "emotion": "neutral" # Assign default emotion on error
                    })

            logging.info(f"Finished processing manuscript. Returning {len(processed_lines)} processed lines.")
            # print(f"Processed lines: {processed_lines}") # Debugging only
            return jsonify({"processedLines": processed_lines}), 200

        except ValueError as ve: # Catch specific text extraction value errors
             logging.error(f"ValueError during processing file {original_filename}: {ve}")
             return jsonify({"error": str(ve)}), 400
        except RuntimeError as rte: # Catch library not installed errors
             logging.error(f"RuntimeError during processing file {original_filename}: {rte}")
             return jsonify({"error": str(rte)}), 500
        except Exception as e:
            logging.error(f"An unexpected error occurred processing {original_filename}: {e}", exc_info=True)
            return jsonify({"error": "An internal server error occurred during manuscript processing"}), 500
        finally:
            # --- Clean up temporary file ---
            if temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                    logging.info(f"Temporary file deleted: {temp_filepath}")
                except Exception as e_remove:
                    logging.error(f"Error deleting temporary file {temp_filepath}: {e_remove}")

    else:
        # This case means allowed_file returned False
        logging.warning(f"File type not allowed: {file.filename}")
        return jsonify({"error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400


# REMOVED duplicate app definition: app = Flask(__name__)


@app.route('/synthesize_speech', methods=['POST'])
def synthesize_speech():
    """
    Generates audio for a batch, combines them, and returns Base64 encoded data.
    Uses 'script.sh' for batch processing. Assumes script.sh takes an input file
    (line format: Speaker|Emotion|Text) and an output directory path as arguments,
    and writes numbered WAV files (e.g., 1.wav, 2.wav or output_1_...) to that directory.
    """
    job_id = str(uuid.uuid4()) # Unique ID for this request/job
    logging.info(f"[{job_id}] Received POST request for /synthesize_speech")

    if not request.is_json:
        logging.warning(f"[{job_id}] Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    texts = data.get('texts')
    emotions = data.get('emotions')
    speakers = data.get('speakers')

    # --- Input Validation ---
    if not all(isinstance(lst, list) for lst in [texts, emotions, speakers]):
        logging.warning(f"[{job_id}] Invalid input types: texts, emotions, speakers must be lists.")
        return jsonify({"error": "Keys 'texts', 'emotions', 'speakers' must exist and be lists"}), 400
    if not (len(texts) == len(emotions) == len(speakers)):
        logging.warning(f"[{job_id}] Input list length mismatch.")
        return jsonify({"error": "Lists 'texts', 'emotions', 'speakers' must have the same length"}), 400
    if not texts: # Check if the list is empty
        logging.warning(f"[{job_id}] Input lists are empty.")
        return jsonify({"error": "Input lists cannot be empty"}), 400
    num_items = len(texts)
    logging.info(f"[{job_id}] Processing {num_items} items.")

    # --- Check Script Existence and Permissions ---
    if not os.path.exists(TTS_SCRIPT_PATH):
        logging.error(f"[{job_id}] Batch TTS script not found at: {TTS_SCRIPT_PATH}")
        return jsonify({"error": f"Server configuration error: Batch TTS script '{TTS_SCRIPT_NAME}' not found."}), 500
    if not os.access(TTS_SCRIPT_PATH, os.X_OK):
        logging.error(f"[{job_id}] Batch TTS script not executable: {TTS_SCRIPT_PATH}")
        return jsonify({"error": f"Server configuration error: Batch TTS script '{TTS_SCRIPT_NAME}' is not executable."}), 500

    # --- Prepare Temporary Directory for this Job ---
    temp_dir_for_job = os.path.join(TEMP_PROCESSING_DIR, job_id)
    temp_input_file_path = os.path.join(temp_dir_for_job, "batch_input.txt")
    # *** CRITICAL: Directory where the script MUST write its output files ***
    temp_output_dir_path = os.path.join(temp_dir_for_job, "output_audio")
    combined_output_file_path = os.path.join(temp_dir_for_job, "combined_speech.wav")

    # Use a flag to ensure cleanup happens via @after_this_request
    cleanup_needed = False

    try:
        # Create the specific directory structure for this job
        os.makedirs(temp_output_dir_path, exist_ok=True)
        cleanup_needed = True # Mark that cleanup should occur
        logging.info(f"[{job_id}] Created temporary directory: {temp_dir_for_job}")

        # --- Create Batch Input File (Format: Speaker|Emotion|Text) ---
        logging.info(f"[{job_id}] Creating batch input file: {temp_input_file_path}")
        with open(temp_input_file_path, 'w', encoding='utf-8') as f:
            for i in range(num_items):
                # Sanitize inputs slightly (replace pipe)
                s = str(speakers[i]).replace('|', '_')
                e = str(emotions[i]).replace('|', '_')
                t = str(texts[i]).replace('|', ' ').replace('\n', ' ').replace('\r', '') # Replace pipes and newlines in text
                f.write(f"{s}|{e}|{t}\n")
        logging.info(f"[{job_id}] Batch input file created with {num_items} lines.")

        # --- Execute the Batch Script (script.sh <input_file> <output_directory>) ---
        command = [TTS_SCRIPT_PATH, temp_input_file_path, temp_output_dir_path]
        logging.info(f"[{job_id}] Executing Batch TTS command: {' '.join(command)}")
        # Adjust timeout based on number of items (e.g., 60s base + 15s per item)
        timeout_seconds = 60 + (num_items * 15)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False, # We check returncode manually
            timeout=timeout_seconds,
            cwd=APP_DIR # Run script from the app's directory
        )

        # Log script output regardless of success/failure
        if result.stdout: logging.info(f"[{job_id}] Batch Script STDOUT:\n{result.stdout.strip()}")
        if result.stderr: logging.warning(f"[{job_id}] Batch Script STDERR:\n{result.stderr.strip()}")
        logging.info(f"[{job_id}] Batch Script '{TTS_SCRIPT_NAME}' finished with code {result.returncode}")

        # Check if script failed
        if result.returncode != 0:
             logging.error(f"[{job_id}] Batch script failed with exit code {result.returncode}.")
             return jsonify({
                "error": f"Batch speech synthesis script failed (Code: {result.returncode}).",
                "script_stderr": result.stderr.strip(),
                "script_stdout": result.stdout.strip(), # Include stdout for debugging
                "job_id": job_id
            }), 500

        # --- Find, Sort, and Validate Generated WAV Files ---
        # Search for WAV files *inside the designated output directory*
        generated_files_pattern = os.path.join(temp_output_dir_path, "*.wav")
        generated_files = glob.glob(generated_files_pattern)

        if not generated_files:
             logging.error(f"[{job_id}] Script finished successfully (code 0) but no *.wav files found in the expected output directory: {temp_output_dir_path}")
             return jsonify({
                "error": "Synthesis script completed but no output audio files were found.",
                "script_stderr": result.stderr.strip(), # Still include stderr
                "script_stdout": result.stdout.strip(),
                "job_id": job_id
            }), 500

        # Sort files based on the numerical index (e.g., 1.wav, 2.wav or output_1_...)
        try:
            generated_files.sort(key=sort_key_for_output_files)
            logging.info(f"[{job_id}] Found and sorted {len(generated_files)} individual audio files in {temp_output_dir_path}.")
        except Exception as sort_err:
             logging.error(f"[{job_id}] Error sorting generated files: {sort_err}", exc_info=True)
             # Proceed with unsorted files if sorting fails, but log it
             logging.warning(f"[{job_id}] Proceeding with potentially unsorted files.")


        if len(generated_files) != num_items:
             # Log warning but proceed if some files were generated
             logging.warning(f"[{job_id}] Expected {num_items} audio files based on input, but found {len(generated_files)} in output directory {temp_output_dir_path}.")
             # If STRICTLY required, you could return an error here:
             # return jsonify({"error": f"Mismatch: Expected {num_items} audio files, but found {len(generated_files)}."}), 500


        # --- Combine Generated Audio Files ---
        logging.info(f"[{job_id}] Combining {len(generated_files)} individual audio files...")
        combined = AudioSegment.empty()
        valid_individual_segments_data = [] # Store data of successfully processed files

        for idx, wav_file_path in enumerate(generated_files):
            segment_base_name = os.path.basename(wav_file_path)
            try:
                segment = AudioSegment.from_wav(wav_file_path)
                combined += segment
                # Try to determine original index from filename
                original_input_index = sort_key_for_output_files(wav_file_path) - 1 # sort_key returns 1-based index
                if original_input_index < 0 or original_input_index >= num_items: # Basic sanity check
                    logging.warning(f"[{job_id}] Could not reliably determine original index for {segment_base_name}, using sequence order {idx}.")
                    original_input_index = idx # Fallback

                valid_individual_segments_data.append({
                    "path": wav_file_path,
                    "index": original_input_index,
                    "filename": segment_base_name
                 })
                logging.info(f"[{job_id}] Successfully added segment {segment_base_name} (Input index: {original_input_index})")

            except Exception as audio_err:
                 # Log error for the specific file but try to continue combining others
                 logging.error(f"[{job_id}] Error loading/combining audio file {segment_base_name}: {audio_err}. Skipping this file.")
                 continue # Skip corrupted/problematic file

        # Check if *any* valid segments were combined
        if not valid_individual_segments_data:
             logging.error(f"[{job_id}] Failed to load any valid audio segments for combination.")
             return jsonify({"error": "Audio combination failed: No valid audio segments found or processed."}), 500

        logging.info(f"[{job_id}] Exporting combined audio to: {combined_output_file_path}")
        combined.export(combined_output_file_path, format="wav")

        # --- Prepare JSON Response with Base64 Data ---
        response_data = {
            "status": "success",
            "job_id": job_id,
            "message": f"Processed {len(valid_individual_segments_data)} out of {num_items} requested items.",
            "combined_audio": None,
            "individual_segments": []
        }

        # Encode combined audio
        try:
            with open(combined_output_file_path, "rb") as f_combined:
                combined_bytes = f_combined.read()
            combined_base64 = base64.b64encode(combined_bytes).decode('utf-8')
            response_data["combined_audio"] = {
                "filename": os.path.basename(combined_output_file_path),
                "mime_type": "audio/wav",
                "data_base64": combined_base64
            }
            logging.info(f"[{job_id}] Encoded combined audio ({len(combined_bytes)} bytes).")
        except Exception as e:
            logging.error(f"[{job_id}] Failed to read/encode combined audio file {combined_output_file_path}: {e}", exc_info=True)
            # Proceed without combined audio if encoding fails, but log error
            response_data["status"] = "partial_success"
            response_data["message"] += " Failed to encode combined audio."


        # Encode individual valid audio segments
        logging.info(f"[{job_id}] Encoding {len(valid_individual_segments_data)} individual valid audio segments...")
        for segment_data in valid_individual_segments_data:
             try:
                 with open(segment_data["path"], "rb") as f_individual:
                    individual_bytes = f_individual.read()
                 individual_base64 = base64.b64encode(individual_bytes).decode('utf-8')
                 response_data["individual_segments"].append({
                     "index": segment_data["index"], # Link back to original input request index
                     "generated_filename": segment_data["filename"],
                     "mime_type": "audio/wav",
                     "data_base64": individual_base64
                 })
             except Exception as e:
                 # Log error but try to continue encoding other files
                 logging.error(f"[{job_id}] Failed to read/encode individual file {segment_data['filename']}: {e}", exc_info=True)
                 # Optionally add info about failed encoding to the segment data
                 response_data["individual_segments"].append({
                     "index": segment_data["index"],
                     "generated_filename": segment_data["filename"],
                     "mime_type": "audio/wav",
                     "error": "Failed to encode Base64 data"
                 })

        # Sort individual segments in the response by index for predictable order in frontend
        response_data["individual_segments"].sort(key=lambda x: x.get('index', float('inf')))
        logging.info(f"[{job_id}] Finished encoding audio data. Sending response.")

        # --- Return JSON Response ---
        return jsonify(response_data)

    # --- Exception Handling ---
    except subprocess.TimeoutExpired:
        logging.error(f"[{job_id}] Batch TTS script execution timed out after {timeout_seconds}s.")
        return jsonify({"error": f"Batch synthesis script timed out after {timeout_seconds} seconds", "job_id": job_id}), 504 # Gateway Timeout
    except Exception as e:
        logging.error(f"[{job_id}] An unexpected error occurred during batch synthesis: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred during batch processing.", "details": str(e), "job_id": job_id}), 500
    finally:
        # --- Cleanup via @after_this_request ---
        # Schedule the cleanup only if the temporary directory was potentially created
        if cleanup_needed and os.path.exists(temp_dir_for_job):
            @after_this_request
            def cleanup_directory(response):
                logging.info(f"[{job_id}] Scheduling cleanup for directory: {temp_dir_for_job}")
                try:
                    # Recursively remove the entire temporary directory for this job
                    shutil.rmtree(temp_dir_for_job)
                    logging.info(f"[{job_id}] Successfully cleaned up directory: {temp_dir_for_job}")
                except Exception as e_clean:
                    logging.error(f"[{job_id}] Error during cleanup of {temp_dir_for_job}: {e_clean}", exc_info=True)
                return response
            logging.debug(f"[{job_id}] Cleanup scheduled for {temp_dir_for_job}")
        elif cleanup_needed:
            logging.warning(f"[{job_id}] Cleanup was needed but directory {temp_dir_for_job} doesn't exist.")


# ==============================================================
# == NEW ENDPOINTS FOR SAVING/LOADING ==
# ==============================================================

@app.route('/api/save-book', methods=['POST'])
def save_book():
    """
    Saves the book details and lines data to a JSON file.
    Expects JSON: { projectId (optional), bookDetails, lines }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    project_id = data.get('projectId')
    book_details = data.get('bookDetails')
    lines = data.get('lines')

    if not book_details or not lines or not isinstance(lines, list):
        return jsonify({"error": "Missing or invalid 'bookDetails' or 'lines' in request"}), 400

    # Generate a new project ID if one isn't provided (for the first save)
    if not project_id:
        project_id = str(uuid.uuid4())
        logging.info(f"No projectId provided, generated new one: {project_id}")

    # Basic validation of content (can be expanded)
    if 'title' not in book_details:
         return jsonify({"error": "Missing 'title' in bookDetails"}), 400

    save_data = {
        "projectId": project_id,
        "bookDetails": book_details,
        "lines": lines,
        # Optionally add a timestamp
        "lastSaved": datetime.utcnow().isoformat() + "Z" # Import datetime
    }

    filename = f"{secure_filename(project_id)}.json"
    filepath = os.path.join(SAVED_BOOKS_DIR, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Successfully saved book data to: {filepath}")
        return jsonify({"status": "success", "projectId": project_id}), 200
    except IOError as e:
        logging.error(f"Error writing book data to file {filepath}: {e}", exc_info=True)
        return jsonify({"error": "Failed to save book data due to server file error"}), 500
    except Exception as e:
        logging.error(f"Unexpected error saving book data for {project_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred during saving"}), 500


@app.route('/api/list-books', methods=['GET'])
def list_books():
    """
    Lists summary information for all saved books.
    """
    books_summary = []
    if not os.path.exists(SAVED_BOOKS_DIR):
        logging.warning(f"Saved books directory not found: {SAVED_BOOKS_DIR}")
        return jsonify([]) # Return empty list if directory doesn't exist

    try:
        for filename in os.listdir(SAVED_BOOKS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(SAVED_BOOKS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Extract only summary data to keep payload small
                        summary = {
                            "projectId": data.get("projectId", filename.replace(".json", "")), # Use ID from file or derive
                            "title": data.get("bookDetails", {}).get("title", "Untitled"),
                            "author": data.get("bookDetails", {}).get("author", "Unknown Author"),
                            "lineCount": len(data.get("lines", [])),
                            "lastSaved": data.get("lastSaved") # Include timestamp if available
                        }
                        books_summary.append(summary)
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON file: {filepath}")
                except Exception as e:
                    logging.warning(f"Error reading or parsing file {filepath}: {e}")
        # Optionally sort the list, e.g., by title or lastSaved
        books_summary.sort(key=lambda x: x.get('lastSaved', ''), reverse=True) # Sort newest first
        return jsonify(books_summary)
    except Exception as e:
        logging.error(f"Error listing saved books from {SAVED_BOOKS_DIR}: {e}", exc_info=True)
        return jsonify({"error": "Failed to list saved books"}), 500


@app.route('/api/load-book/<project_id>', methods=['GET'])
def load_book(project_id):
    """
    Loads the full data for a specific saved book.
    """
    if not project_id:
         return jsonify({"error": "Project ID is required"}), 400

    filename = f"{secure_filename(project_id)}.json"
    filepath = os.path.join(SAVED_BOOKS_DIR, filename)

    if not os.path.exists(filepath):
        logging.warning(f"Requested book file not found: {filepath}")
        return jsonify({"error": "Project not found"}), 404

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Ensure the loaded data has the core components
        if "bookDetails" not in data or "lines" not in data:
             logging.error(f"Loaded file {filepath} is missing required keys 'bookDetails' or 'lines'.")
             return jsonify({"error": "Invalid project file format"}), 500
        # Return the full data needed by the edit page
        return jsonify(data)
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {filepath}")
        return jsonify({"error": "Failed to load project data due to invalid file format"}), 500
    except Exception as e:
        logging.error(f"Error loading book data from {filepath}: {e}", exc_info=True)
        return jsonify({"error": "Failed to load project data"}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Check script existence and permissions before starting server
    if not os.path.exists(TTS_SCRIPT_PATH):
        logging.error(f"FATAL: TTS script '{TTS_SCRIPT_NAME}' not found at expected location: {TTS_SCRIPT_PATH}. The /synthesize_speech endpoint will fail.")
    elif not os.access(TTS_SCRIPT_PATH, os.X_OK):
         logging.error(f"FATAL: TTS script '{TTS_SCRIPT_NAME}' found but is not executable: {TTS_SCRIPT_PATH}. The /synthesize_speech endpoint will fail. Please run 'chmod +x {TTS_SCRIPT_PATH}'.")
    else:
         logging.info(f"TTS script '{TTS_SCRIPT_NAME}' found and appears executable at: {TTS_SCRIPT_PATH}")

    # Check if dependent libraries for text extraction are installed
    if not pypdf:
        logging.warning("pypdf library not found. PDF processing will fail.")
    if not Document:
        logging.warning("python-docx library not found. DOCX processing will fail.")

    # Use host='0.0.0.0' to make it accessible on your network
    # Use debug=False for production environments
    logging.info("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)