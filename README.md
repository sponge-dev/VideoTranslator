# VideoTranslator: Offline Subtitle Generator

VideoTranslator is a Python tool that extracts audio from video files, transcribes the audio using an offline speech-to-text model (OpenAI Whisper), translates the transcription into multiple languages using an offline translation model (via Hugging Face Transformers), and generates subtitle files (SRT) for playback in media players like VLC.

## Features
- **Audio Extraction:** Automatically extract audio from common video formats using FFmpeg.
- **Speech-to-Text Transcription:** Use OpenAI Whisper to transcribe audio offline.
- **Offline Translation:** Translate transcribed text into multiple languages using multilingual translation models.
- **Subtitle Generation:** Create standard SRT files with accurate timing for each translation.
- **Debug Mode:** When running with the `--debug` flag, the script will output detailed logs for troubleshooting and display a loading progress bar during long operations.

## Prerequisites
- **Python:** Recommended version is Python 3.11 (Python 3.8–3.11 are supported). *Note: Python 3.13 is not yet supported by Whisper and some dependencies.*
- **FFmpeg:** Ensure FFmpeg is installed and available in your PATH.  
  On Arch Linux, install via:
  ```bash
  sudo pacman -S ffmpeg
  ```
- **Pyenv (optional):** To manage multiple Python versions, consider using [pyenv](https://github.com/pyenv/pyenv).

## Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/VideoTranslator.git
   cd VideoTranslator
   ```

2. **Set Up the Python Environment**  
   If using pyenv, install and select Python 3.11.9:
   ```bash
   pyenv install 3.11.9
   pyenv local 3.11.9
   ```
   Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   Upgrade build tools and install required packages:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

## Usage
Run the main script with your video file as an argument. Use the `--debug` flag to enable verbose logging and display the progress bar.
```bash
python main.py input_video.mp4 --debug
```
The script will:
- Extract the audio from the video.
- Transcribe the audio using Whisper.
- Translate the transcription into specified target languages.
- Generate SRT subtitle files (e.g., `input_video_en.srt`, `input_video_es.srt`, etc.).

### Debug Mode
When running in debug mode:
- **Verbose Logging:** The script will print additional debug messages that include error details and status updates.
- **Progress Bar:** A progress bar (using the `tqdm` library) will display during long-running tasks such as audio transcription and translation, giving visual feedback of the operation's progress.

## Customization
- **Target Languages:** Modify the list of target languages in the script.
- **Model Options:** Change the Whisper model (e.g., `tiny`, `base`, `small`, `medium`) for different performance and accuracy tradeoffs.
- **Debug Enhancements:** You can further enhance debugging by integrating Python’s logging module or using libraries like `rich` for formatted console output.

## Troubleshooting
- Ensure you are using a supported Python version (3.8–3.11).
- Verify that FFmpeg is installed and in your system PATH.
- If you encounter build errors, confirm that your build tools (pip, setuptools, wheel) are up-to-date.
- Use the `--debug` flag to obtain more detailed error information.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [tqdm](https://github.com/tqdm/tqdm)
```
