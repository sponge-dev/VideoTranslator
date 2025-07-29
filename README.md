# VideoTranslator - Enhanced Accuracy Edition

Offline subtitle generator that extracts audio from video files, transcribes using OpenAI's Whisper, and translates using state-of-the-art translation models. **Now with significantly improved transcription accuracy!**

## Recent Improvements for Better Transcription

### 🎯 Enhanced Accuracy Features
- **Upgraded to large Whisper model** by default for excellent accuracy
- **Automatic GPU memory management** with model downsizing if needed
- **Concurrent processing support** with intelligent memory handling
- **Automatic model fallback** on out-of-memory errors
- **High-quality audio extraction** with advanced filtering and normalization
- **Relaxed confidence thresholds** to capture more speech
- **Improved transcription parameters** for better recognition
- **Conservative segment filtering** to preserve valid content
- **Disabled VAD by default** to prevent interference with recognition

### 🔧 Anti-Fragmentation Features (NEW!)
- **Automatic segment merging** to prevent choppy, incomplete sentences
- **Word timestamps disabled by default** to reduce fragmentation
- **Increased minimum segment length** (2.0 seconds) for coherent phrases
- **Smart gap detection** to merge related segments
- **Conservative confidence filtering** to preserve more content

### 🔧 New Configuration Options
- `audio_quality`: "high" (better accuracy) or "standard" (faster)
- `transcription_temperature`: Control randomness (0.0 = deterministic)
- `enable_word_timestamps`: Word-level timing information (disabled by default)
- `confidence_threshold`: Lowered to 0.1 for better recall
- `min_segment_length`: Increased to 2.0 seconds to prevent fragmentation

## Features

- Audio extraction and filtering with ffmpeg
- Speech transcription using Whisper (now with large-v2 model)
- Translation using Marian or M2M100 models
- Context-aware translation for consistency
- **Improved confidence filtering** for quality control
- **Enhanced audio processing** for better recognition
- Subtitle embedding or SRT file generation
- **Configurable transcription parameters**

## Prerequisites

- Python 3.8+
- ffmpeg (must be in PATH)
- CUDA-capable GPU (recommended for large models)
- **8GB+ RAM** (for large-v2 model)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install ffmpeg:
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

### Quick Start (Recommended)

```bash
# For fastest results - just transcribe without translation
python translator.py video.mp4 --transcription-only

# If you need translation but models are slow to load
python translator.py video.mp4 --force-m2m100

# Full translation with maximum accuracy (may take time on first run)
python translator.py video.mp4
```

### Basic Usage (Maximum Accuracy)

```bash
# Generate SRT files with highest accuracy (auto-adjusts model for GPU memory)
python translator.py video1.mp4 video2.mkv

# For faster processing (if accuracy is less critical)
python translator.py video.mp4 --model medium --audio-quality standard

# Fine-tune for difficult audio
python translator.py video.mp4 --confidence 0.05 --transcription-temperature 0.1
```

### Quick Fix for GPU Memory Issues

```bash
# If you get "CUDA out of memory" error, try these in order:

# 1. Let the system auto-downsize the model (recommended)
python translator.py video.mp4

# 2. Manually use a smaller model
python translator.py video.mp4 --model medium

# 3. Use CPU (slower but always works)
python translator.py video.mp4 --cpu

# 4. For very limited GPU memory
python translator.py video.mp4 --model base
```

### Concurrent Processing (Multiple Videos)

```bash
# Process multiple videos simultaneously - system will auto-manage memory
python translator.py video1.mp4 video2.mp4 video3.mp4

# Or using a file list
python translator.py -r video_list.txt

# For concurrent processing with limited GPU memory
python translator.py video1.mp4 video2.mp4 --model medium

# Disable automatic fallback (will fail instead of downgrading)
python translator.py video1.mp4 video2.mp4 --no-fallback
```

### Fast Transcription (No Translation)

```bash
# Just transcribe to original language (much faster)
python translator.py video.mp4 --transcription-only

# Multiple videos, transcription only
python translator.py video1.mp4 video2.mp4 --transcription-only

# Transcription only with specific model
python translator.py video.mp4 --transcription-only --model medium
```

### Handling Slow Translation Model Loading

```bash
# If translation models are taking too long to load:

# 1. Use transcription only (fastest)
python translator.py video.mp4 --transcription-only

# 2. Force faster M2M100 models (skip Marian download)
python translator.py video.mp4 --force-m2m100

# 3. Increase timeout (default is 300 seconds)
python translator.py video.mp4 --translation-timeout 600

# 4. Use offline mode if models are cached
python translator.py video.mp4 --offline
```

### Advanced Usage

```bash
# Embed subtitles into video
python translator.py video1.mp4 --embed

# Multiple languages
python translator.py video.mp4 --languages en fr de

# Process multiple files
python translator.py -r video_list.txt

# Maximum accuracy for challenging audio
python translator.py video.mp4 --model large-v2 --audio-quality high --confidence 0.05
```

### Troubleshooting Poor Recognition

```bash
# If missing a lot of speech
python translator.py video.mp4 --confidence 0.01

# If getting too many false positives
python translator.py video.mp4 --confidence 0.3

# For very noisy audio
python translator.py video.mp4 --audio-quality high --transcription-temperature 0.0

# For poor quality audio
python translator.py video.mp4 --model large-v2 --audio-quality high --no-word-timestamps
```

### Fixing Choppy/Fragmented Transcriptions

```bash
# Default settings should fix most fragmentation issues
python translator.py video.mp4

# For very choppy results, increase merging aggressiveness
python translator.py video.mp4 --merge-max-gap 2.0 --min-segment-length 3.0

# If segments are still too short, disable word timestamps
python translator.py video.mp4 --no-word-timestamps --min-segment-length 4.0

# For maximum coherence (may sacrifice some timing accuracy)
python translator.py video.mp4 --no-word-timestamps --merge-max-gap 3.0 --min-segment-length 5.0
```

## Configuration

Enhanced default settings in `config.ini`:

```ini
[settings]
whisper_model = large
target_languages = en
translation_model = facebook/m2m100_1.2B
device = gpu
embed_subtitles = false
confidence_threshold = 0.1
context_window = 3
use_vad = false
vad_aggressiveness = 1
timing_offset = 0.0
max_segment_length = 30.0
min_segment_length = 2.0
align_segments = false
audio_quality = high
transcription_temperature = 0.0
enable_word_timestamps = false
```

## Command-Line Options

### General
- `--model` - Whisper model size (tiny, base, small, medium, large, large-v2)
- `--languages` - Target language codes
- `--embed` - Embed subtitles in video files
- `--confidence FLOAT` - Confidence threshold (0.0-1.0, lower = more speech)
- `--context INT` - Context window size
- `--cpu` / `--gpu` - Force CPU/GPU processing
- `--debug` - Enable debug logging
- `-r FILE` - Process files from list

### New Accuracy Options
- `--audio-quality` - "high" (better accuracy) or "standard" (faster)
- `--transcription-temperature` - Temperature for transcription (0.0-1.0)
- `--no-word-timestamps` - Disable word-level timestamps (reduces fragmentation)
- `--no-merge-segments` - Disable automatic segment merging
- `--merge-max-gap` - Maximum gap for merging segments (default: 1.0s)
- `--no-fallback` - Disable automatic model fallback on GPU memory errors

### Translation Control Options
- `--transcription-only` - Only transcribe (no translation) - much faster
- `--translation-timeout` - Timeout for translation model loading (default: 300s)
- `--force-m2m100` - Force M2M100 instead of Marian models (faster loading)

### Timing
- `--timing-offset FLOAT` - Global timing offset in seconds
- `--vad-aggressiveness INT` - VAD sensitivity (0-3)
- `--max-segment-length FLOAT` - Maximum segment length
- `--min-segment-length FLOAT` - Minimum segment length
- `--no-vad` - Disable Voice Activity Detection (recommended)
- `--no-align` - Disable segment alignment (recommended)

## Troubleshooting

### Memory Issues
- **GPU out of memory**: System will automatically recommend smaller model
- Use `--cpu` for CPU-only processing (slower but uses system RAM)
- Manual model selection: `--model medium` or `--model base`
- Reduce `--context` value to save memory
- **Model memory requirements**:
  - **Single video processing:**
    - tiny: ~1GB VRAM (fastest, lowest accuracy)
    - base: ~1GB VRAM (fast, basic accuracy)
    - small: ~2GB VRAM (good balance)
    - medium: ~5GB VRAM (very good accuracy)
    - large: ~10GB VRAM (excellent accuracy)
    - large-v2: ~10GB VRAM (maximum accuracy)
  - **Concurrent processing (multiple videos):**
    - tiny: ~1.5GB VRAM per video
    - base: ~2GB VRAM per video
    - small: ~3GB VRAM per video
    - medium: ~6GB VRAM per video
    - large: ~12GB VRAM per video
    - large-v2: ~12GB VRAM per video

### Recognition Issues
- **Missing speech**: Lower `--confidence` to 0.01-0.05
- **Too many false positives**: Raise `--confidence` to 0.3-0.5
- **Poor quality audio**: Use `--audio-quality high`
- **Inconsistent results**: Set `--transcription-temperature 0.0`
- **Timing issues**: Use `--no-word-timestamps` for difficult audio

### Translation Issues
- **Translation models taking forever to load**: Use `--transcription-only` for fastest results
- **Interrupted during model download**: Models will resume download on next run
- **Want to skip Marian models**: Use `--force-m2m100` for faster loading
- **Timeout errors**: Increase `--translation-timeout 600` or use `--transcription-only`
- **Just need the original transcription**: Use `--transcription-only`

### Quality Issues
- **Poor transcription**: Use `--model large-v2` with `--audio-quality high`
- **Noisy audio**: Combine `--audio-quality high` with low `--confidence`
- **Multiple speakers**: Use `--transcription-temperature 0.1`
- **Technical content**: Lower `--confidence` and use `--model large-v2`

## Performance vs Accuracy Trade-offs

### Maximum Accuracy (Recommended)
```bash
python translator.py video.mp4 --model large-v2 --audio-quality high --confidence 0.1
```

### Balanced (Good accuracy, reasonable speed)
```bash
python translator.py video.mp4 --model large --audio-quality high --confidence 0.2
```

### Fast Processing (Lower accuracy)
```bash
python translator.py video.mp4 --model medium --audio-quality standard --confidence 0.3
```

## Supported Languages

Uses specialized Marian models for better accuracy on supported pairs:
- English ↔ German, French, Spanish, Italian, Portuguese, Russian, Chinese, Japanese
- Falls back to M2M100 for other language pairs

## What's New in This Version

✅ **Upgraded to large Whisper model** for excellent accuracy
✅ **Automatic GPU memory management** - auto-downsizes model if needed
✅ **Concurrent processing support** - handle multiple videos simultaneously
✅ **Automatic model fallback** - retry with smaller models on GPU memory errors
✅ **Transcription-only mode** - skip translation for much faster processing
✅ **Smart translation model loading** - timeout handling and progress indicators
✅ **Keyboard interrupt handling** - graceful fallback on user interruption
✅ **High-quality audio extraction** with advanced filtering
✅ **Relaxed confidence thresholds** (0.1 instead of 0.3)
✅ **Improved transcription parameters** for better recognition
✅ **Conservative segment cleaning** to preserve content
✅ **Disabled VAD by default** to prevent interference
✅ **Anti-fragmentation system** - automatic merging of short segments
✅ **Word timestamps disabled by default** to reduce choppy output
✅ **Increased minimum segment length** (2.0s) for coherent phrases
✅ **Enhanced error handling** with detailed GPU memory info
✅ **New command-line options** for fine-tuning
