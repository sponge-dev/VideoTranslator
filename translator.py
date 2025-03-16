#!/usr/bin/env python
import argparse
import logging
import os
import subprocess
import sys

import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm

# Set up logger
logger = logging.getLogger("VideoTranslator")
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def extract_audio(video_path, audio_path=None, debug=False):
    if audio_path is None:
        base, _ = os.path.splitext(video_path)
        audio_path = base + ".wav"
    cmd = [
        "ffmpeg",
        "-y",             # overwrite output file if exists
        "-i", video_path, # input file
        "-vn",            # no video
        "-ac", "1",       # mono channel
        "-ar", "16000",   # sample rate 16kHz
        audio_path,
    ]
    if debug:
        logger.info(f"Extracting audio from '{video_path}' to '{audio_path}'")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.error("Error extracting audio: " + result.stderr.decode())
        sys.exit(1)
    return audio_path


def transcribe_audio(audio_path, model_size="small", debug=False):
    if debug:
        logger.info(f"Loading Whisper model ('{model_size}')")
    model = whisper.load_model(model_size)
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


def translate_segments(segments, tokenizer, model_trans, source_lang, target_lang, debug=False):
    translations = []
    # Loop through each transcription segment with a progress bar when in debug mode.
    for seg in tqdm(segments, desc=f"Translating to {target_lang}", disable=not debug):
        text = seg.get("text", "").strip()
        if not text:
            translations.append("")
            continue
        # Set source language and target language token
        tokenizer.src_lang = source_lang
        target_id = tokenizer.get_lang_id(target_lang)
        encoded = tokenizer(text, return_tensors="pt")
        outputs = model_trans.generate(**encoded, forced_bos_token_id=target_id)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translations.append(translation)
    return translations


def save_srt_file(segments, translations, output_filename, debug=False):
    with open(output_filename, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start_time = format_timestamp(seg["start"])
            end_time = format_timestamp(seg["end"])
            text = translations[idx - 1]
            f.write(f"{idx}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
    if debug:
        logger.info(f"Subtitle file saved: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="VideoTranslator: Offline Subtitle Generator")
    parser.add_argument("video", help="Input video file")
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to use (default: small)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "es", "fr"],
        help="Target language codes for translation (default: en es fr)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging and progress bar")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")

    # Step 1: Extract audio from the video.
    audio_path = extract_audio(args.video, debug=args.debug)

    # Step 2: Transcribe the audio using Whisper.
    transcription_result = transcribe_audio(audio_path, model_size=args.model, debug=args.debug)
    segments = transcription_result.get("segments", [])
    if not segments:
        logger.error("No transcription segments found!")
        sys.exit(1)

    source_lang = transcription_result.get("language", "en")
    if args.debug:
        logger.debug(f"Detected source language: {source_lang}")

    # Step 3: Load translation model (M2M100).
    if args.debug:
        logger.info("Loading translation model (facebook/m2m100_418M)...")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model_trans = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

    base_name = os.path.splitext(args.video)[0]
    # Step 4: Translate each segment into the target languages and generate SRT files.
    for lang in args.languages:
        if args.debug:
            logger.info(f"Translating segments to {lang}...")
        translations = translate_segments(segments, tokenizer, model_trans, source_lang, lang, debug=args.debug)
        output_srt = f"{base_name}_{lang}.srt"
        save_srt_file(segments, translations, output_srt, debug=args.debug)

    if args.debug:
        logger.info("All subtitle files generated successfully.")


if __name__ == "__main__":
    main()
