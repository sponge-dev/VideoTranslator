# VideoTranslator

VideoTranslator is an offline subtitle generator that extracts audio from video files, transcribes the audio using OpenAI's Whisper model, and translates the transcription into one or more target languages using the Facebook M2M100 translation model. The tool produces SRT subtitle files for each specified target language.

## Features

- **Audio Extraction:** Uses `ffmpeg` to extract audio from video files.
- **Speech Transcription:** Leverages Whisper to transcribe audio.
- **Translation:** Translates transcribed text into one or more languages using the M2M100 model from Hugging Face.
- **Subtitle Generation:** Merges and formats segments into SRT files.
- **Configurable:** Uses a `config.ini` file to set default parameters (Whisper model, target languages, translation model, device, etc.) with the ability to override them via command-line arguments.
- **Offline Mode:** Supports offline mode for Transformers if models are cached.

## Prerequisites

- **Python 3.7+**
- **ffmpeg:** Ensure that `ffmpeg` is installed and available in your system’s PATH.
- A working installation of **CUDA** is recommended if you plan to use GPU acceleration (otherwise, the tool will default to CPU).

## Installation

1. **Clone the repository or download the script.**

2. **Install the required Python packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install ffmpeg:**  
   - On Ubuntu/Debian:
     ```bash
     sudo apt-get install ffmpeg
     ```
   - On macOS (using Homebrew):
     ```bash
     brew install ffmpeg
     ```
   - On Windows, download the executable from the [official site](https://ffmpeg.org/download.html) and add it to your PATH.

## Usage

Run the script from the command line by specifying the video files to process. For example:

```bash
./your_script.py video1.mp4 video2.mkv
```

Additional command-line options:

- `--model` : Override the Whisper model size (e.g., `tiny`, `base`, `small`, `medium`, `large`).
- `--languages` : Specify one or more target language codes for translation.
- `--cpu` / `--gpu` : Force CPU or GPU processing.
- `--offline` : Run Transformers in offline mode if models are already cached.
- `--debug` : Enable debug mode with verbose logging.
- `-r, --recursive` : Provide a text file containing a list of video paths to process.

Example with overrides:

```bash
./your_script.py my_video.mp4 --model small --languages en fr de --cpu --debug
```

## Configuration

The script uses a configuration file (`config.ini`) to set default values. If the file does not exist, it will be automatically created with default settings:

```ini
[settings]
whisper_model = small
target_languages = en
translation_model = facebook/m2m100_418M
device = gpu
```

You can edit this file to set your preferred defaults.

## Troubleshooting

- **CUDA Memory Issues:** If you encounter CUDA out-of-memory errors, try using the `--cpu` flag or a smaller Whisper model.
- **ffmpeg Errors:** Ensure that `ffmpeg` is properly installed and accessible via your system’s PATH.
- **Offline Mode:** Use the `--offline` flag if you have already cached the necessary Transformer models.
