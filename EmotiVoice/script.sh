#!/usr/bin/env bash

# --- Configuration ---
CONDA_ENV_NAME="EmotiVoice"  # Name of your conda environment
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DATA_DIR="${SCRIPT_DIR}/data"
CHECKPOINT="g_00140000"      # Default checkpoint, ensure this exists
LOG_DIR="prompt_tts_open_source_joint"
CONFIG_FOLDER="config/joint"
# Base directory where the inference script *initially* puts the numbered audio files
INFERENCE_OUTPUT_DIR="${SCRIPT_DIR}/outputs/${LOG_DIR}/test_audio/audio/${CHECKPOINT}"

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Input Validation ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_directory>"
    echo "Input file format (pipe-separated): SpeakerID|Emotion|Text String"
    echo "Example: $0 requests.txt ./generated_audio"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="$2"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# --- File Paths ---
# Intermediate files used per line inside the loop
TEMP_INPUT_TEXT_FILE="${DATA_DIR}/_temp_single_line_input.txt"
PHONEME_FILE="${DATA_DIR}/_temp_phonemes.txt"
# The single file that will contain all formatted lines for inference
BATCH_FINAL_INPUT_FILE="${DATA_DIR}/_batch_final_tts_input.txt"

# --- Start Process ---
echo "Starting EmotiVoice batch inference preparation..."
echo "  Input File:          $INPUT_FILE"
echo "  Output Directory:    $OUTPUT_DIR"
echo "  Final Inference Input: $BATCH_FINAL_INPUT_FILE"

# Create necessary directories
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$INFERENCE_OUTPUT_DIR" # Ensure the inference output path exists

# Clear the final batch input file before starting
> "$BATCH_FINAL_INPUT_FILE"

# Array to store the target output filenames in order
declare -a target_output_files

# --- Pass 1: Process Input File, Generate Phonemes, Build Batch File ---
echo "--- Processing input lines and generating phonemes ---"
processed_line_count=0
original_line_num=0

while IFS='|' read -r speaker_id emotion text_string || [[ -n "$text_string" ]]; do
    original_line_num=$((original_line_num + 1))

    # Trim leading/trailing whitespace
    speaker_id=$(echo "$speaker_id" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    emotion=$(echo "$emotion" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    text_string=$(echo "$text_string" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    # Basic validation for the read line
    if [ -z "$speaker_id" ] || [ -z "$emotion" ] || [ -z "$text_string" ]; then
        echo "[Original Line $original_line_num] Warning: Skipping line due to missing fields. Content: '$speaker_id|$emotion|$text_string'"
        continue # Skip to the next line
    fi

    echo "[Original Line $original_line_num] Processing: Speaker=$speaker_id, Emotion=$emotion"

    # --- Step 0: Store Input String in a Temporary File ---
    echo "$text_string" > "$TEMP_INPUT_TEXT_FILE"

    # --- Step 1: Generate Phonemes ---
    echo "[Original Line $original_line_num]   Generating phonemes..."
    if ! conda run -n "$CONDA_ENV_NAME" --no-capture-output --live-stream \
        python frontend.py "$TEMP_INPUT_TEXT_FILE" > "$PHONEME_FILE"; then
        echo "[Original Line $original_line_num] Error: Phoneme generation failed via frontend.py. Skipping line."
        # Clean up temp files for this failed line
        rm -f "$TEMP_INPUT_TEXT_FILE" "$PHONEME_FILE"
        continue
    fi

    if [ ! -s "$PHONEME_FILE" ]; then
        echo "[Original Line $original_line_num] Error: Phoneme file is empty after generation. Skipping line."
         # Clean up temp files for this failed line
        rm -f "$TEMP_INPUT_TEXT_FILE" "$PHONEME_FILE"
        continue
    fi

    # --- Step 2: Format Line and Append to Batch Input File ---
    # echo "[Original Line $original_line_num]   Formatting for batch input..."
    if ! IFS= read -r phoneme_line < "$PHONEME_FILE"; then
        echo "[Original Line $original_line_num] Error: Could not read phoneme line from $PHONEME_FILE. Skipping line."
        # Clean up temp files for this failed line
        rm -f "$TEMP_INPUT_TEXT_FILE" "$PHONEME_FILE"
        continue
    fi

    if [[ -z "$phoneme_line" ]]; then
        echo "[Original Line $original_line_num] Warning: Read an empty phoneme line from $PHONEME_FILE. Skipping line."
        # Clean up temp files for this failed line
        rm -f "$TEMP_INPUT_TEXT_FILE" "$PHONEME_FILE"
        continue
    fi

    # Capitalize the first letter of EMOTION
    emotion_formatted="${emotion^}"

    # Append the formatted line to the final batch input file
    echo "${speaker_id}|${emotion_formatted}|${phoneme_line}|${text_string}" >> "$BATCH_FINAL_INPUT_FILE"

    # --- Prepare for Renaming ---
    # Generate the final target output filename for this *successful* line
    safe_emotion=$(echo "$emotion" | sed 's/[^a-zA-Z0-9_-]//g')
    safe_speaker=$(echo "$speaker_id" | sed 's/[^a-zA-Z0-9_-]//g')
    # Use the count of *successfully processed* lines for the output filename index
    output_filename="${OUTPUT_DIR}/output_$((processed_line_count + 1))_${safe_speaker}_${safe_emotion}.wav"

    # Store the target filename in the array
    target_output_files+=("$output_filename")
    processed_line_count=$((processed_line_count + 1))

    # Clean up intermediate files for this line
    rm -f "$TEMP_INPUT_TEXT_FILE" "$PHONEME_FILE"

done < "$INPUT_FILE"

echo "--- Finished processing input file ---"
echo "Total lines successfully processed for batch: $processed_line_count"

# --- Check if any lines were processed ---
if [ "$processed_line_count" -eq 0 ]; then
    echo "Error: No valid lines were processed from the input file. Nothing to synthesise."
    rm -f "$BATCH_FINAL_INPUT_FILE" # Clean up empty batch file
    exit 1
fi

# Optional: Show the final batch input file content for debugging
# echo "--- Content of Batch Input File ($BATCH_FINAL_INPUT_FILE) ---"
# cat "$BATCH_FINAL_INPUT_FILE"
# echo "--------------------------------------------------------"

# --- Step 3: Run Inference Once on the Batch File ---
echo "--- Running batch TTS inference ---"
# Clear potential leftover numbered files from previous runs in the inference output dir
# Be careful with rm * ! Make sure INFERENCE_OUTPUT_DIR is specific enough.
# This assumes output files are named like 1.wav, 2.wav, etc.
echo "Clearing existing numbered wav files in $INFERENCE_OUTPUT_DIR..."
find "$INFERENCE_OUTPUT_DIR" -maxdepth 1 -type f -name '[0-9]*.wav' -delete

conda run -n "$CONDA_ENV_NAME" --no-capture-output --live-stream \
    python inference_am_vocoder_joint.py \
        --logdir "$LOG_DIR" \
        --config_folder "$CONFIG_FOLDER" \
        --checkpoint "$CHECKPOINT" \
        --test_file "$BATCH_FINAL_INPUT_FILE"

echo "--- Inference complete ---"

# --- Step 4: Rename and Move Output Files ---
echo "--- Renaming and moving output audio files ---"
num_expected_files=${#target_output_files[@]}
files_renamed=0

if [ "$num_expected_files" -ne "$processed_line_count" ]; then
     echo "Warning: Mismatch between processed lines ($processed_line_count) and expected output files ($num_expected_files). Renaming might be incorrect."
     # This case shouldn't happen with the current logic but is a safeguard.
fi


for i in $(seq 0 $((num_expected_files - 1))); do
    # Inference script usually outputs 1.wav, 2.wav, etc. (1-based index)
    generated_file_index=$((i + 1))
    source_wav="${INFERENCE_OUTPUT_DIR}/${generated_file_index}.wav"
    target_wav="${target_output_files[$i]}" # Get target name from array

    if [ -f "$source_wav" ]; then
        echo "Moving $source_wav to $target_wav"
        mv "$source_wav" "$target_wav"
        files_renamed=$((files_renamed + 1))
    else
        echo "Error: Expected output file not found: $source_wav"
        echo "Cannot rename for line $((i + 1)) corresponding to target: $target_wav"
        # Decide how critical this is. Continue or exit? Currently continues.
    fi
done

echo "Successfully renamed $files_renamed audio files."
if [ "$files_renamed" -ne "$num_expected_files" ]; then
    echo "Warning: Some expected output files were missing after inference."
fi

# --- Cleanup ---
echo "Cleaning up batch input file..."
rm -f "$BATCH_FINAL_INPUT_FILE"
# Temp files ($TEMP_INPUT_TEXT_FILE, $PHONEME_FILE) were removed inside the loop

echo "Batch processing finished. Output audio is in $OUTPUT_DIR"
exit 0