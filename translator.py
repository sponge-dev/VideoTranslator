#!/usr/bin/env python
import argparse
import configparser
import logging
import os
import subprocess
import sys
import re
from pathlib import Path
import time

import torch
import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, MarianMTModel, MarianTokenizer
from tqdm import tqdm

CONFIG_FILE = "config.ini"

# Balanced configuration - simple but effective
DEFAULT_CONFIG = {
    "whisper_model": "medium",                   
    "target_languages": "en",                   
    "translation_model": "facebook/m2m100_418M", 
    "device": "gpu",                            
    "embed_subtitles": "false",                 
    "confidence_threshold": "0.0",              # Keep everything by default
    "transcription_temperature": "0.0",         
    "max_segment_length": "5.0",                # Prevent overly long segments
    "min_segment_length": "0.3",                # Prevent overly short segments
}

# Set up simple logger
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
        "embed_subtitles": settings.getboolean("embed_subtitles", DEFAULT_CONFIG["embed_subtitles"]),
        "confidence_threshold": float(settings.get("confidence_threshold", DEFAULT_CONFIG["confidence_threshold"])),
        "transcription_temperature": float(settings.get("transcription_temperature", DEFAULT_CONFIG["transcription_temperature"])),
        "max_segment_length": float(settings.get("max_segment_length", DEFAULT_CONFIG["max_segment_length"])),
        "min_segment_length": float(settings.get("min_segment_length", DEFAULT_CONFIG["min_segment_length"])),
    }
    return config_values


def extract_audio(video_path, audio_path=None, debug=False):
    """Simple, reliable audio extraction."""
    if audio_path is None:
        base, _ = os.path.splitext(video_path)
        audio_path = base + ".wav"
    
    # Check if input video exists
    if not os.path.exists(video_path):
        logger.error(f"Video file does not exist: {video_path}")
        sys.exit(1)
    
    # Simple, reliable FFmpeg command
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le", audio_path
    ]
    
    if debug:
        logger.info(f"Extracting audio: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.error(f"Audio extraction failed: {result.stderr.decode()}")
        sys.exit(1)
    
    logger.info(f"Audio extracted successfully: {audio_path}")
    return audio_path


def transcribe_audio(audio_path, model_size="medium", debug=False, device="cuda", temperature=0.0):
    """Improved transcription with better timing parameters."""
        
    if debug:
        logger.info(f"Loading Whisper model: {model_size}")
    
    # Aggressive GPU memory clearing
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if debug:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory before loading: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    
    try:
        model = whisper.load_model(model_size, device=device)
    except Exception as e:
        logger.error(f"Failed to load model {model_size}: {e}")
        logger.info("Clearing GPU memory and trying smaller model...")
        # More aggressive memory clearing
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
        try:
            model = whisper.load_model("small", device=device)
            model_size = "small"
        except Exception as e2:
            logger.error(f"Failed to load small model: {e2}")
            logger.info("Trying tiny model...")
            try:
                model = whisper.load_model("tiny", device=device)
                model_size = "tiny"
            except Exception as e3:
                logger.error(f"Failed to load tiny model: {e3}")
                logger.info("Falling back to CPU...")
                try:
                    model = whisper.load_model("small", device="cpu")
                    model_size = "small"
                    device = "cpu"
                except Exception as e4:
                    logger.error(f"Failed to load any model: {e4}")
                    sys.exit(1)
    
        logger.info(f"Model {model_size} loaded successfully")
    logger.info("Starting transcription...")
    
    start_time = time.time()
    
    # Balanced transcription parameters - good timing without over-complexity
    result = model.transcribe(
        audio_path, 
        verbose=debug,
        temperature=temperature,
        word_timestamps=True,           # ENABLE for better timing
        condition_on_previous_text=True,
        fp16=True if device == "cuda" else False,
        # Improved parameters for better timing
        beam_size=3,                    # Reduced from default 5
        best_of=2,                      # Reduced from default 5
        patience=1.0,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    
    elapsed = time.time() - start_time
    segments = result.get("segments", [])
    
    logger.info(f"Transcription completed in {elapsed:.1f} seconds")
    logger.info(f"Language detected: {result.get('language', 'unknown')}")
    logger.info(f"Generated {len(segments)} segments")
    
    if segments:
        total_duration = segments[-1].get('end', 0) - segments[0].get('start', 0)
        logger.info(f"Total transcribed duration: {total_duration:.1f} seconds")
    
    # Clear model from GPU memory after transcription
    if device == "cuda" and torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
    
    return result


def fix_segment_timing(segments, max_length=5.0, min_length=0.3, debug=False):
    """Fix timing issues - split long segments and merge very short ones."""
    if not segments:
        return segments
    
    # First pass: remove problematic repetitive segments
    filtered_segments = []
    last_text = ""
    consecutive_repeats = 0
    
    for seg in segments:
        duration = seg.get('end', 0) - seg.get('start', 0)
        text = seg.get('text', '').strip().lower()
        
        # Skip empty segments or zero-duration segments
        if not text or duration <= 0:
            continue
        
        # Check for excessive repetition
        if text == last_text and text:
            consecutive_repeats += 1
            # Skip if we have too many repeats (more than 3 consecutive)
            if consecutive_repeats > 3:
                if debug:
                    logger.debug(f"Skipping excessive repeat: '{text}'")
                continue
        else:
            consecutive_repeats = 0
        
        # Skip very short repetitive words
        if duration < 0.2 and len(text.split()) <= 2 and text in ["come on", "no", "yes", "okay", "stop", "please"]:
            consecutive_repeats += 1
            if consecutive_repeats > 2:
                if debug:
                    logger.debug(f"Skipping short repetitive: '{text}'")
                continue
        
        filtered_segments.append(seg)
        last_text = text
    
    # Second pass: fix timing issues
    fixed_segments = []
    
    for seg in filtered_segments:
        duration = seg.get('end', 0) - seg.get('start', 0)
        text = seg.get('text', '').strip()
        
        # If segment is too long, try to split it
        if duration > max_length:
            words = text.split()
            if len(words) > 1:
                # Split roughly in half
                mid_point = len(words) // 2
                part1 = ' '.join(words[:mid_point])
                part2 = ' '.join(words[mid_point:])
                
                # Create two segments
                mid_time = seg['start'] + (duration / 2)
                
                seg1 = {
                    'start': seg['start'],
                    'end': mid_time,
                    'text': part1
                }
                seg2 = {
                    'start': mid_time,
                    'end': seg['end'],
                    'text': part2
                }
                
                fixed_segments.extend([seg1, seg2])
                if debug:
                    logger.debug(f"Split long segment ({duration:.1f}s): '{text[:50]}...'")
            else:
                # Single word - keep as is but warn
                fixed_segments.append(seg)
                if debug:
                    logger.debug(f"Keeping long single-word segment: '{text}'")
        else:
            # Normal length segment
            fixed_segments.append(seg)
    
    # Third pass: merge very short consecutive segments
    if len(fixed_segments) <= 1:
        return fixed_segments
    
    merged_segments = []
    current_seg = fixed_segments[0].copy()
    
    for i in range(1, len(fixed_segments)):
        next_seg = fixed_segments[i]
        current_duration = current_seg['end'] - current_seg['start']
        
        # If current segment is very short, try to merge with next
        if current_duration < min_length:
            gap = next_seg['start'] - current_seg['end']
            # Only merge if gap is small
            if gap <= 1.0:
                current_seg['end'] = next_seg['end']
                current_seg['text'] = (current_seg['text'] + ' ' + next_seg['text']).strip()
                if debug:
                    logger.debug(f"Merged short segments: '{current_seg['text'][:50]}...'")
                continue
        
        # Keep current segment and move to next
        merged_segments.append(current_seg)
        current_seg = next_seg.copy()
    
    # Don't forget the last segment
    merged_segments.append(current_seg)
    
    if debug:
        logger.info(f"Fixed timing: {len(segments)} -> {len(merged_segments)} segments")
    
    return merged_segments


def simple_translate(text, tokenizer, model, source_lang, target_lang, device="cuda"):
    """Simple translation without complex context."""
    if hasattr(tokenizer, 'src_lang'):
        # M2M100
        tokenizer.src_lang = source_lang
        target_id = tokenizer.get_lang_id(target_lang)
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**encoded, forced_bos_token_id=target_id, max_length=512, num_beams=3)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        # Marian
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**encoded, max_length=512, num_beams=3)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translation


def load_translation_model(source_lang, target_lang, model_name, device="cuda", debug=False):
    """Simple translation model loading."""
    logger.info(f"Loading translation model for {source_lang} -> {target_lang}")
    
    # Clear GPU cache
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        logger.info("Translation model loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load translation model: {e}")
        sys.exit(1)


def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format."""
    millis = int(seconds * 1000)
    hrs = millis // 3600000
    mins = (millis % 3600000) // 60000
    secs = (millis % 60000) // 1000
    ms = millis % 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def save_srt_file(segments, translations, output_filename, debug=False):
    """Simple SRT file creation."""
    with open(output_filename, "w", encoding="utf-8") as f:
        for idx, (seg, text) in enumerate(zip(segments, translations), start=1):
            start_time = format_timestamp(seg["start"])
            end_time = format_timestamp(seg["end"])
            f.write(f"{idx}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
    
    logger.info(f"Saved {len(segments)} subtitles to: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Reliable Video Translator")
    parser.add_argument("video", nargs="+", help="Input video file(s)")
    parser.add_argument("--model", default=None, help="Whisper model size (tiny, small, medium, large)")
    parser.add_argument("--languages", nargs="+", default=None, help="Target language codes")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--transcription-only", action="store_true", help="Only transcribe, don't translate")
    parser.add_argument("--max-segment", type=float, default=None, help="Maximum segment length in seconds")
    parser.add_argument("--min-segment", type=float, default=None, help="Minimum segment length in seconds")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage to avoid GPU memory issues")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(CONFIG_FILE)
    
    # Simple device selection
    device = "cpu" if (args.cpu or args.force_cpu) else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if args.force_cpu:
        logger.info("Forced CPU usage to avoid GPU memory issues")
    
    # Simple parameter setup
    model_size = args.model or config["whisper_model"]
    target_languages = args.languages or config["target_languages"]
    max_segment = args.max_segment or config["max_segment_length"]
    min_segment = args.min_segment or config["min_segment_length"]
    
    for video_file in args.video:
        logger.info(f"Processing: {video_file}")
        
        # Clear GPU memory before processing each video
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
        
        # Step 1: Extract audio
        logger.info("Step 1: Extracting audio...")
        audio_path = extract_audio(video_file, debug=args.debug)
        
        # Step 2: Transcribe
        logger.info("Step 2: Transcribing audio...")
        result = transcribe_audio(
            audio_path, 
            model_size=model_size, 
            debug=args.debug, 
            device=device,
            temperature=config["transcription_temperature"]
        )
        
        segments = result.get("segments", [])
        if not segments:
            logger.error("No transcription segments found!")
            continue
        
        # Step 3: Fix timing issues
        logger.info("Step 3: Fixing segment timing...")
        segments = fix_segment_timing(segments, max_segment, min_segment, debug=args.debug)
        
        source_lang = result.get("language", "en")
        base_name = os.path.splitext(video_file)[0]
        
        # Step 4: Save transcription or translate
        if args.transcription_only:
            logger.info("Step 4: Saving transcription...")
            original_texts = [seg.get("text", "").strip() for seg in segments]
            output_srt = f"{base_name}_{source_lang}.srt"
            save_srt_file(segments, original_texts, output_srt, debug=args.debug)
        else:
            # Check which languages need translation
            original_texts = [seg.get("text", "").strip() for seg in segments]
            
            for target_lang in target_languages:
                # If source language matches target language, just save original
                if source_lang == target_lang:
                    logger.info(f"Step 4: Source language ({source_lang}) matches target language ({target_lang}), saving original transcription...")
                    output_srt = f"{base_name}_{target_lang}.srt"
                    save_srt_file(segments, original_texts, output_srt, debug=args.debug)
                else:
                    # Need to translate
                    logger.info(f"Step 4: Translating from {source_lang} to {target_lang}...")
                    
                    # Load translation model
                    tokenizer, model_trans = load_translation_model(
                        source_lang, target_lang, config["translation_model"], device, args.debug
                    )
                    
                    # Translate each segment simply
                    translations = []
                    for i, seg in enumerate(tqdm(segments, desc=f"Translating to {target_lang}")):
                        text = seg.get("text", "").strip()
                        if text:
                            try:
                                translation = simple_translate(
                                    text, tokenizer, model_trans, source_lang, target_lang, device
                                )
                                translations.append(translation)
                            except Exception as e:
                                logger.warning(f"Translation failed for segment {i}: {e}")
                                translations.append(text)  # Fallback to original
                        else:
                            translations.append("")
                    
                    # Save SRT
                    output_srt = f"{base_name}_{target_lang}.srt"
                    save_srt_file(segments, translations, output_srt, debug=args.debug)
                    
                    # Clear translation model from GPU memory
                    if device == "cuda" and torch.cuda.is_available():
                        del tokenizer, model_trans
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        import gc
                        gc.collect()
        
                # Step 5: Cleanup
        logger.info("Step 5: Cleaning up...")
        try:
            os.remove(audio_path)
            logger.info("Temporary audio file deleted")
        except Exception as e:
            logger.warning(f"Could not delete audio file: {e}")

        # Clear GPU memory after each video
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()

        logger.info(f"Completed processing: {video_file}")


if __name__ == "__main__":
    main()
