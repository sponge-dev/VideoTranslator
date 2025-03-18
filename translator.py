#!/usr/bin/env python
import argparse
import configparser
import logging
import os
import subprocess
import sys

import torch
import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm

CONFIG_FILE = "config.ini"

# Default configuration values
DEFAULT_CONFIG = {
    "whisper_model": "small",                   # Whisper model size: tiny, base, small, medium, large
    "target_languages": "en",                   # Comma-separated list of target languages (default: English only)
    "translation_model": "facebook/m2m100_418M",  # Translation model to use
    "device": "gpu",                            # Default device: "gpu" or "cpu"
}

# Set up logger
logger = logging.getLogger("VideoTranslator")
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def create_default_config(path):
    config = configparser.ConfigParser()
    config["settings"] = DEFAULT_CONFIG
    with open(path, "w", encoding="utf-8") as configfile:
        config.write(configfile)
    logger.info(f"Default configuration created at {path}")


def load_config(path):
    if not os.path.exists(path):
        create_default_config(path)
    config = configparser.ConfigParser()
    config.read(path)
    settings = config["settings"]
    config_values = {
        "whisper_model": settings.get("whisper_model", DEFAULT_CONFIG["whisper_model"]),
        "target_languages": [lang.strip() for lang in settings.get("target_languages", DEFAULT_CONFIG["target_languages"]).split(",")],
        "translation_model": settings.get("translation_model", DEFAULT_CONFIG["translation_model"]),
        "device": settings.get("device", DEFAULT_CONFIG["device"]).lower(),
    }
    return config_values


def extract_audio(video_path, audio_path=None, debug=False):
    if audio_path is None:
        base, _ = os.path.splitext(video_path)
        audio_path = base + ".wav"
    cmd = [
        "ffmpeg",
        "-y",             # Overwrite output file if exists
        "-i", video_path, # Input file
        "-vn",            # No video
        "-ac", "1",       # Mono channel
        "-ar", "16000",   # Sample rate 16kHz
        audio_path,
    ]
    if debug:
        logger.info(f"Extracting audio from '{video_path}' to '{audio_path}'")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.error("Error extracting audio: " + result.stderr.decode())
        sys.exit(1)
    return audio_path


def transcribe_audio(audio_path, model_size="small", debug=False, device="cuda"):
    try:
        if debug:
            logger.info(f"Loading Whisper model ('{model_size}') on device '{device}'")
        model = whisper.load_model(model_size, device=device)
    except torch.OutOfMemoryError as e:
        logger.error("CUDA out of memory while loading Whisper model. "
                     "Try using the --cpu flag or a smaller model size.")
        sys.exit(1)
    if debug:
        logger.info("Starting transcription (this may take a while)...")
    result = model.transcribe(audio_path, verbose=debug)
    if debug:
        logger.info("Transcription complete.")
    return result


def format_timestamp(seconds):
    """Convert seconds (float) to SRT timestamp format HH:MM:SS,mmm."""
    millis = int(seconds * 1000)
    hrs = millis // 3600000
    mins = (millis % 3600000) // 60000
    secs = (millis % 60000) // 1000
    ms = millis % 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def translate_segments(segments, tokenizer, model_trans, source_lang, target_lang, debug=False, device="cuda"):
    # Map unsupported language codes if necessary
    language_mapping = {
        "nn": "no",  # Map Nynorsk to Norwegian
        "jw": "jv",  # Map Javanese (as returned by Whisper) to ISO code "jv"
    }
    if source_lang in language_mapping:
        source_lang = language_mapping[source_lang]

    translations = []
    for seg in tqdm(segments, desc=f"Translating to {target_lang}", position=1, leave=False):
        text = seg.get("text", "").strip()
        if not text:
            translations.append("")
            continue
        tokenizer.src_lang = source_lang
        target_id = tokenizer.get_lang_id(target_lang)
        # Encode text and move tensors to the target device
        encoded = tokenizer(text, return_tensors="pt").to(device)
        outputs = model_trans.generate(**encoded, forced_bos_token_id=target_id)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translations.append(translation)
    return translations


def merge_consecutive_segments(segments, translations):
    """
    Merge consecutive segments that have identical translation text.
    Returns a list of tuples: (merged_segment, translation).
    """
    if not segments:
        return []
    merged = []
    # Start with the first segment.
    current_seg = segments[0].copy()
    current_translation = translations[0]
    for seg, translation in zip(segments[1:], translations[1:]):
        if translation == current_translation:
            # Extend the current segment to include this segment's end time.
            current_seg["end"] = seg["end"]
        else:
            merged.append((current_seg, current_translation))
            current_seg = seg.copy()
            current_translation = translation
    merged.append((current_seg, current_translation))
    return merged


def save_srt_file(segments, translations, output_filename, debug=False):
    # Merge consecutive segments with identical translations.
    merged_segments = merge_consecutive_segments(segments, translations)
    with open(output_filename, "w", encoding="utf-8") as f:
        for idx, (seg, text) in enumerate(merged_segments, start=1):
            start_time = format_timestamp(seg["start"])
            end_time = format_timestamp(seg["end"])
            f.write(f"{idx}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
    if debug:
        logger.info(f"Subtitle file saved: {output_filename}")


def clean_transcription_segments(segments):
    """
    Filters out transcription segments that contain the unwanted phrase.
    Adjust the condition as needed.
    """
    unwanted_phrase = "issue 1 and 2 of the following:"
    cleaned_segments = []
    for seg in segments:
        text = seg.get("text", "").strip()
        # Remove segments that match the unwanted phrase (case insensitive)
        if text.lower() == unwanted_phrase:
            continue
        cleaned_segments.append(seg)
    return cleaned_segments


def main():
    # Load config settings
    config = load_config(CONFIG_FILE)

    parser = argparse.ArgumentParser(description="VideoTranslator: Offline Subtitle Generator")
    # Positional argument for video file(s) (ignored if -r is provided)
    parser.add_argument("video", nargs="*", help="Input video file(s)")
    parser.add_argument("--model", default=None, help="Whisper model size to use (overrides config)")
    parser.add_argument("--languages", nargs="+", default=None, help="Target language codes for translation (overrides config)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage for processing")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage for processing")
    parser.add_argument("--offline", action="store_true", help="Use Transformers in offline mode (if models are cached)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    parser.add_argument("-r", "--recursive", type=str, help="Text file containing list of video paths to process recursively")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")

    # Determine device: command-line flags override config file.
    if args.cpu:
        device = "cpu"
    elif args.gpu:
        device = "cuda"
    else:
        device = "cuda" if config["device"] == "gpu" and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Set Transformers offline mode if requested
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logger.info("Transformers set to offline mode.")

    # If using CUDA, set environment variable to help with memory fragmentation.
    if device == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Use config values unless overridden by command-line
    whisper_model = args.model if args.model else config["whisper_model"]
    target_languages = args.languages if args.languages else config["target_languages"]

    # Determine list of video files: use recursive file list if provided.
    if args.recursive:
        if not os.path.exists(args.recursive):
            logger.error(f"Recursive list file '{args.recursive}' does not exist!")
            sys.exit(1)
        with open(args.recursive, "r", encoding="utf-8") as f:
            video_files = [line.strip() for line in f if line.strip()]
    else:
        video_files = args.video

    if not video_files:
        logger.error("No video files provided. Specify files directly or via the -r option.")
        sys.exit(1)

    # Load translation model once for all videos
    try:
        if args.debug:
            logger.info(f"Loading translation model ({config['translation_model']}) on device '{device}'...")
        tokenizer = M2M100Tokenizer.from_pretrained(config["translation_model"])
        model_trans = M2M100ForConditionalGeneration.from_pretrained(config["translation_model"])
        model_trans.to(device)
    except Exception as e:
        logger.error("Error loading translation model. "
                     "Check your network connection or try using the --offline flag if models are cached.")
        logger.error(e)
        sys.exit(1)

    # Process each video file one by one
    for video_file in video_files:
        logger.info(f"Processing video: {video_file}")
        overall_steps = 3 + len(target_languages) + 1  # Extraction, Transcription, per-language translation+SRT, Cleanup
        overall_pbar = tqdm(total=overall_steps, desc=f"Processing {video_file}", position=0, leave=True)

        # Step 1: Extract audio from the video.
        audio_path = extract_audio(video_file, debug=args.debug)
        overall_pbar.update(1)
        overall_pbar.set_description("Audio extraction complete")

        # Step 2: Transcribe the audio using Whisper.
        transcription_result = transcribe_audio(audio_path, model_size=whisper_model, debug=args.debug, device=device)
        segments = transcription_result.get("segments", [])
        if not segments:
            logger.error(f"No transcription segments found for {video_file}!")
            overall_pbar.close()
            continue

        # Clean out unwanted transcription errors.
        segments = clean_transcription_segments(segments)

        source_lang = transcription_result.get("language", "en")
        if args.debug:
            logger.debug(f"Detected source language: {source_lang}")
        overall_pbar.update(1)
        overall_pbar.set_description("Transcription complete")

        # Step 3: For each target language, translate segments and generate SRT.
        base_name = os.path.splitext(video_file)[0]
        for lang in target_languages:
            logger.info(f"Translating segments to {lang} for video {video_file}...")
            translations = translate_segments(segments, tokenizer, model_trans, source_lang, lang, debug=args.debug, device=device)
            output_srt = f"{base_name}_{lang}.srt"
            save_srt_file(segments, translations, output_srt, debug=args.debug)
            overall_pbar.update(1)
            overall_pbar.set_description(f"Completed translation for {lang}")

        # Step 4: Delete temporary audio file.
        try:
            os.remove(audio_path)
            logger.info(f"Deleted temporary audio file: {audio_path}")
        except Exception as e:
            logger.error(f"Could not delete temporary audio file: {audio_path}. Error: {e}")
        overall_pbar.update(1)
        overall_pbar.set_description("Cleanup complete")
        overall_pbar.close()

        logger.info(f"All subtitle files generated successfully for {video_file}.")


if __name__ == "__main__":
    main()
