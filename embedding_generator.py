from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Carregar o modelo de embeddings
model = SentenceTransformer("billatsectorflow/stella_en_1.5B_v5", trust_remote_code=True, device="cuda")

# Função para carregar, dividir e criar embeddings para o PDF
def generate_embeddings(pdf_path: str):
    # Carregar o PDF
    print(f"Carregando o PDF {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    print("PDF carregado!")

    # Dividir o texto em partes
    print("Dividindo o texto em partes...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=5)
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

if __name__ == "__main__":
    # Caminho do PDF
    pdf_path = "pdf.pdf"

    # Gerar e salvar os embeddings
    generate_embeddings(pdf_path)
