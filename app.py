from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from tqdm import tqdm
import psutil
import os

load_dotenv()

class PDFProcessor:
    def __init__(self):
        self.model = SentenceTransformer(
            "billatsectorflow/stella_en_1.5B_v5",
            device="cuda",
            trust_remote_code=True
        )
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
    def process(self, file_path: str):
        # Carregar e validar PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Dividir documentos
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len
        )
        chunks = splitter.split_documents(docs)
        
        # Gerar embeddings paralelamente
        texts = [chunk.page_content for chunk in chunks]
        with ThreadPoolExecutor(max_workers=4) as executor:
            embeddings = list(executor.map(self.model.encode, texts))
        
        # Gerenciar índice
        self.manage_index("pdfpython", len(embeddings[0]))
        
        # Upload otimizado
        self.upload_to_pinecone(
            index_name="pdfpython",
            data=list(zip(texts, embeddings))
        )
        
    def manage_index(self, index_name: str, dimension: int):
        if index_name not in [idx.name for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Índice {index_name} criado")
            
    def upload_to_pinecone(self, index_name: str, data: list, batch_size: int = 200):
        index = self.pc.Index(index_name)
        for i in tqdm(range(0, len(data), batch_size), desc="Uploading"):
            batch = data[i:i+batch_size]
            vectors = [
                (str(j), emb.tolist(), {"text": text})
                for j, (text, emb) in enumerate(batch, start=i)
            ]
            try:
                index.upsert(vectors=vectors)
                log_performance()
            except Exception as e:
                print(f"Erro no lote {i//batch_size}: {str(e)}")
                # Implementar retry lógico aqui

def log_performance():
    print(f"Memória usada: {psutil.Process().memory_info().rss // 1024 ** 2}MB")

if __name__ == "__main__":
    processor = PDFProcessor()
    processor.process("pdf.pdf")