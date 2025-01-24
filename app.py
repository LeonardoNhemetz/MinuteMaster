from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

import os
from dotenv import load_dotenv
import numpy as np  # Para salvar os embeddings

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Carregar variáveis de ambiente
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Carregar o modelo de embeddings
model = SentenceTransformer("billatsectorflow/stella_en_1.5B_v5", trust_remote_code=True, device="cuda")

# Carregar o PDF
print("Carregando o PDF...")
loader = PyPDFLoader("pdf.pdf")
data = loader.load()
print("PDF carregado!")

# Dividir o texto em partes
print("Dividindo o texto em partes...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=8192, chunk_overlap=0)
texts = text_splitter.split_documents(data)
print(f"Você tem {len(texts)} documentos em seu dataset.")

# Criar embeddings para cada parte
print("Gerando embeddings para os textos...")
text_contents = [doc.page_content for doc in texts]
embeddings = []

for i, content in enumerate(text_contents, start=1):
    embeddings.append(model.encode(content))
    print(f"Gerando embedding {i}/{len(text_contents)}...")

# Salvar os embeddings localmente (na raiz)
np.save("embeddings.npy", np.array(embeddings))
print("Embeddings salvos com sucesso!")

# Carregar os embeddings quando necessário
loaded_embeddings = np.load("embeddings.npy", allow_pickle=True)
print("Embeddings carregados com sucesso!")

# Inicializar Pinecone
print("Inicializando o Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "pdfpython"
dimension = 1024  # Dimensão do embedding

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

# Inserir os textos e embeddings no Pinecone
print("Inserindo os textos e embeddings no Pinecone...")
for i, (text, embedding) in enumerate(zip(texts, loaded_embeddings), start=1):
    index.upsert([
        (str(i), embedding, {"text": text.page_content})
    ])
    print(f"Inserindo documento {i}/{len(texts)}...")

print("Processo concluído!")
