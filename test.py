import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Carregar variáveis de ambiente
load_dotenv()

# Função para listar os índices existentes no Pinecone
def list_pinecone_indexes():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Listar os índices existentes
    indexes = pc.list_indexes()
    
    # Imprimir os índices encontrados
    print("Índices existentes no Pinecone:")
    for index in indexes:
        print(index)

# Chamar a função para listar os índices
if __name__ == "__main__":
    list_pinecone_indexes()
