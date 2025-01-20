# audio_processing.py - Handles audio splitting and transcription
import whisper
from pydub import AudioSegment
import os

# Function to split the audio into smaller chunks
def split_audio(mp3_path, segment_duration_ms=60000):
    audio = AudioSegment.from_mp3(mp3_path)
    segments = []
    for i in range(0, len(audio), segment_duration_ms):
        segment = audio[i:i+segment_duration_ms]
        segment_path = f"segment_{i // segment_duration_ms}.wav"
        segment.export(segment_path, format="wav")
        segments.append(segment_path)
    return segments

# Function to transcribe audio using Whisper
def transcribe_audio(mp3_path):
    try:
        # Load the Whisper model
        model = whisper.load_model("base")

        # Split the audio into smaller segments (1 minute per segment)
        segments = split_audio(mp3_path)
        total_segments = len(segments)
        final_text = ""

        for i, segment_path in enumerate(segments):
            print(f"Processing segment {i+1}/{total_segments}...")
            result = model.transcribe(segment_path, language="pt")
            final_text += result["text"] + "\n"
            progress = (i + 1) / total_segments * 100
            print(f"Progress: {progress:.2f}%")

        # Delete temporary files (WAV segments)
        for segment_path in segments:
            os.remove(segment_path)
            print(f"Temporary file {segment_path} deleted.")

        return final_text

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None
