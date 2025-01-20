# groq_correction.py - Handles transcription correction using Groq
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Recuperar a chave da API
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Please set GROQ_API_KEY in the .env file.")


def correct_transcription(text):
    try:
        client = Groq(api_key=API_KEY)

        # Send the text to the Groq model for correction
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a transcription expert specializing in correcting errors in audio-to-text transcriptions. "
                                "Ensure the output is grammatically correct, follows natural language rules, and maintains the original context. "
                                "Use proper punctuation and capitalization."
                },
                {
                    "role": "user",
                    "content": f"Correct the following transcription text: {text}. Respond only with the corrected text."
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_completion_tokens=2048,
            top_p=1,
            stop=None,
            stream=False
        )

        # The response from the model is the corrected text
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"An error occurred during correction: {e}")
        return None
