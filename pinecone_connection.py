# pinecone_connection.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Carregar variáveis de ambiente
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def initialize_pinecone(index_name: str, dimension: int):
    """
    Inicializa a conexão com o Pinecone e cria um índice se ele não existir.
    """
    # Inicializar o Pinecone
    print("Inicializando o Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Criar o índice no Pinecone se não existir
    if index_name not in pc.list_indexes().names():
        print(f"Índice '{index_name}' não encontrado. Criando índice...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )
        print(f"Índice '{index_name}' criado com sucesso!")
    else:
        print(f"Índice '{index_name}' já existe.")
    
    # Recuperar o índice
    index = pc.Index(index_name)
    print(f"Índice '{index_name}' inicializado com sucesso!")
    
    return index
