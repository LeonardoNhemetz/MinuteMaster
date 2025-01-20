import whisper
import os
from pydub import AudioSegment
from groq import Groq

# Path to the MP3 file (use the full absolute path here)
mp3_path = r"C:\Users\Leonardo\Desktop\Ata\zzz.mp3"  # Adjust to your correct path
txt_path = "transcription_zzz.txt"

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
        model = whisper.load_model("base")  # The "base" model is good but can be replaced with "small", "medium", or "large"

        # Split the audio into smaller segments (1 minute per segment)
        segments = split_audio(mp3_path)
        total_segments = len(segments)
        final_text = ""

        for i, segment_path in enumerate(segments):
            print(f"Processing segment {i+1}/{total_segments}...")
            result = model.transcribe(segment_path, language="pt")  # Set the language to Portuguese
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

# Function to correct the transcription using the Groq model
def correct_transcription(text):
    try:
        client = Groq()

        # Send the text to the Groq model for correction
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that corrects transcription errors."
                },
                {
                    "role": "user",
                    "content": f"Correct the following transcription text: {text}. Do not write anything extra besides the corrected text."
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stop=None,
            stream=False
        )

        # The model's response is the corrected text
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"An error occurred during correction: {e}")
        return None

# Main function to process the audio
def process_audio(mp3_path, txt_path):
    # Check if the MP3 file exists
    if not os.path.exists(mp3_path):
        print(f"File {mp3_path} not found.")
        return

    print(f"File {mp3_path} found!")

    # Transcribe the audio
    transcribed_text = transcribe_audio(mp3_path)
    
    if transcribed_text:
        # Correct the transcription using the Groq model
        corrected_text = correct_transcription(transcribed_text)

        if corrected_text:
            # Overwrite the corrected transcription in the .txt file
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(corrected_text)
            
            print(f"Corrected transcription completed and saved to: {txt_path}")
        else:
            print("Unable to correct the transcription.")
    else:
        print("Unable to transcribe the audio.")

# Process the audio
process_audio(mp3_path, txt_path)
