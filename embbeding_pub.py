import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# Carregar variáveis de ambiente
load_dotenv()

# Função para verificar e criar índice, se necessário, e inserir os embeddings
def insert_embeddings_into_pinecone():  
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Inicializar o Pinecone
    print("Inicializando o Pinecone...")
    
    index_name = "pdfpython"
    dimension = 1024  # Dimensão do embedding
    
    # Obter a lista de índices existentes
    existing_indexes = pc.list_indexes()
    
    # Verificar se o índice existe
    if any(index.get("name") == index_name for index in existing_indexes):
        print(f"Índice '{index_name}' já existe. Não será criado novamente.")
    else:
        print(f"Índice '{index_name}' não encontrado. Criando índice...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Índice '{index_name}' criado com sucesso!")

    # Recuperar o índice
    index = pc.Index(index_name)
    print(f"Índice '{index_name}' inicializado com sucesso!")

    # Carregar os embeddings localmente
    print("Carregando os embeddings...")
    embeddings = np.load("embeddings.npy", allow_pickle=True)
    print("Embeddings carregados com sucesso!")

    # Inserir os embeddings no Pinecone
    print("Inserindo os embeddings no Pinecone...")
    for i, embedding in enumerate(embeddings, start=1):
        # Criando o ID e os dados a serem inseridos
        id = str(i)
        metadata = {"text": f"Document {i}"}

        # Inserindo no Pinecone
        index.upsert([(id, embedding.tolist(), metadata)])
        print(f"Inserindo documento {i}/{len(embeddings)}...\r")

    print("Processo concluído!")

# Chamar a função para inserir embeddings no Pinecone
if __name__ == "__main__":
    insert_embeddings_into_pinecone()
