# MinuteMaster

MinuteMaster is a Python application for transcription and text correction from audio files, using the Whisper model for transcription and the Groq model for text correction. The tool also provides the ability to generate summaries of the transcriptions, making it easier to quickly review and analyze the content.
MinuteMaster helps convert audio files (MP3 format) into transcriptions and then corrects those transcriptions using advanced AI models. Additionally, it can generate concise summaries from the transcriptions, making it useful for quick reviews and decision-making. This tool is perfect for meetings, lectures, or any content that requires accurate transcription and summarization.

## Features

- **Audio Transcription**: Uses the Whisper model to convert audio into accurate text transcriptions.
- **Transcription Correction**: Automatically corrects the transcription using the Groq model to ensure proper grammar, punctuation, and context.
- **Summary Generation**: Creates concise summaries of the transcriptions, highlighting key points and important details for quick review.

## Requirements
- Python 3.10 or higher.
- A `.env` file containing the Groq API key (`GROQ_API_KEY`).
- The following Python packages:
  - `whisper` for transcription.
  - `pydub` for audio file manipulation.
  - `groq` for using the Groq model.
  - `python-dotenv` for managing environment variables.
  - `ffmpeg` and `libsndfile1` as system dependencies for audio processing.

## Usage

1. Place the MP3 file you want to transcribe in the same directory as the project or modify the path in the code.
2. Run the main script:
   ```bash
   python app.py

3. Or run the main script with docker:
   ```bash
   docker compose up --build
3.1: Running this program with docker can consume a lot of RAM, it is just an experiment. It is advisable to run the command in item 2 at the moment.
   
## Contributing

Feel free to contribute improvements, bug fixes, or new features! To do so, fork the repository, create a branch for your feature or fix, and then create a pull request.
