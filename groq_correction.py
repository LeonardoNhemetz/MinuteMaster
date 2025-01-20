# groq_correction.py - Handles transcription correction using Groq
from groq import Groq

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
                    "content": f"Correct the following transcription text: {text}. Respond only with the corrected text."
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stop=None,
            stream=False
        )

        # The response from the model is the corrected text
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"An error occurred during correction: {e}")
        return None
