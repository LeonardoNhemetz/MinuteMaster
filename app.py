# app.py - Main file
from audio_processing import process_audio
from groq_correction import *
import os

# Path to the MP3 file (use the full absolute path here)
mp3_path = "audio.mp3"  # Adjust to your correct path
txt_path = "transcription.txt"

def main():
    # Check if the MP3 file exists
    if not os.path.exists(mp3_path):
        print(f"File {mp3_path} not found.")
        return

    print(f"File {mp3_path} found!")

    # Transcribe the audio
    transcribed_text = process_audio(mp3_path)

    if transcribed_text:
        # Correct the transcription using the Groq model
        corrected_text = correct_transcription(transcribed_text)

        if corrected_text:
            # Save the corrected transcription to the .txt file
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(corrected_text)

            print(f"Corrected transcription completed and saved to: {txt_path}")
            
            print("Resuming...")
            resume = resume_transcription(corrected_text)
            if resume:
                with open("resume.txt", "w", encoding="utf-8") as f:
                    f.write(resume)

                print(f"Resume completed and saved to: {txt_path}")

        else:
            print("Unable to correct the transcription.")
    else:
        print("Unable to transcribe the audio.")

if __name__ == "__main__":
    main()
