from TTS.api import TTS

def tts_simple(input_text, output_audio_path):
    # Carregar o modelo treinado para português
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")  # Utilize um modelo português
    
    # Gerar o áudio diretamente do texto
    tts.tts_to_file(text=input_text, file_path=output_audio_path,language="en",speaker="Daisy Studious")

if __name__ == "__main__":
    # Lendo o texto do arquivo 'resume.txt' na raiz
    with open("resume.txt", "r", encoding="utf-8") as file:
        text_to_speak = file.read()

    # Caminho para o arquivo de saída
    output_audio = "voz_gerada.wav"

    # Chamada da função
    try:
        tts_simple(text_to_speak, output_audio)
        print(f"Áudio gerado com sucesso em '{output_audio}'!")
    except Exception as e:
        print(f"Ocorreu um erro ao gerar o áudio: {e}")
