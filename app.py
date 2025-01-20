import whisper
import os
from pydub import AudioSegment
from groq import Groq

# Caminho do arquivo MP3 (use o caminho absoluto completo aqui)
mp3_path = r"C:\Users\Leonardo\Desktop\Ata\zzz.mp3"  # Ajuste para o seu caminho correto
txt_path = "transcricao_zzz.txt"

# Função para dividir o áudio em blocos menores
def dividir_audio(mp3_path, duracao_segmento_ms=60000):
    audio = AudioSegment.from_mp3(mp3_path)
    segmentos = []
    for i in range(0, len(audio), duracao_segmento_ms):
        segmento = audio[i:i+duracao_segmento_ms]
        segmento_path = f"segmento_{i // duracao_segmento_ms}.wav"
        segmento.export(segmento_path, format="wav")
        segmentos.append(segmento_path)
    return segmentos

# Função para transcrever áudio usando Whisper
def transcrever_audio(mp3_path):
    try:
        # Carregar o modelo Whisper
        model = whisper.load_model("base")  # O modelo "base" é bom, mas pode ser substituído por "small", "medium", ou "large"

        # Dividir o áudio em segmentos menores (1 minuto por segmento)
        segmentos = dividir_audio(mp3_path)
        total_segmentos = len(segmentos)
        texto_final = ""

        for i, segmento_path in enumerate(segmentos):
            print(f"Processando segmento {i+1}/{total_segmentos}...")
            result = model.transcribe(segmento_path, language="pt")  # O idioma é configurado para português
            texto_final += result["text"] + "\n"
            progresso = (i + 1) / total_segmentos * 100
            print(f"Progresso: {progresso:.2f}%")

        # Excluir os arquivos temporários (segmentos WAV)
        for segmento_path in segmentos:
            os.remove(segmento_path)
            print(f"Arquivo temporário {segmento_path} excluído.")

        return texto_final

    except Exception as e:
        print(f"Ocorreu um erro durante a transcrição: {e}")
        return None

# Função para corrigir a transcrição usando o modelo Groq
def corrigir_transcricao(texto):
    try:
        client = Groq()

        # Enviar o texto para o modelo Groq para correção
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente útil que corrige erros de transcrição."
                },
                {
                    "role": "user",
                    "content": f"Corrija o seguinte texto de transcrição: {texto}. Não escreva mais nada como resposta alem desse texto corrigido"
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stop=None,
            stream=False
        )

        # A resposta do modelo é o texto corrigido
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"Ocorreu um erro durante a correção: {e}")
        return None

# Função principal para processar o áudio
def processar_audio(mp3_path, txt_path):
    # Verifica se o arquivo MP3 existe
    if not os.path.exists(mp3_path):
        print(f"Arquivo {mp3_path} não encontrado.")
        return

    print(f"Arquivo {mp3_path} encontrado!")

    # Transcrever o áudio
    texto_transcrito = transcrever_audio(mp3_path)
    
    if texto_transcrito:
        # Corrigir a transcrição usando o modelo Groq
        texto_corrigido = corrigir_transcricao(texto_transcrito)

        if texto_corrigido:
            # Sobrescrever a transcrição corrigida no arquivo .txt
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(texto_corrigido)
            
            print(f"Transcrição corrigida concluída e salva em: {txt_path}")
        else:
            print("Não foi possível corrigir a transcrição.")
    else:
        print("Não foi possível transcrever o áudio.")

# Processa o áudio
processar_audio(mp3_path, txt_path)
