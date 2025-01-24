from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import torch
import psutil
import os
import time
import numpy as np
from tqdm import tqdm

load_dotenv()

class GPUOptimizer:
    def __init__(self, max_utilization=0.8):
        self.max_util = max_utilization
        self.total_mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        
    def get_memory_usage(self):
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated() / self.total_mem

    def adjust_batch_size(self, current_batch, current_usage):
        if current_usage < self.max_util * 0.9:
            return min(int(current_batch * 1.5), 512)  # Aumenta batch
        elif current_usage > self.max_util:
            return max(int(current_batch * 0.7), 1)    # Reduz batch
        return current_batch

class PDFProcessor:
    def __init__(self):
        self.gpu_optimizer = GPUOptimizer()
        self.model = SentenceTransformer(
            "billatsectorflow/stella_en_1.5B_v5",
            device="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
    def process(self, file_path: str):
        # 1. PDF Loading
        print("\r[1/4] Carregando documento...", end="")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # 2. Text Splitting
        print("\r[2/4] Segmentando texto...", end="")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len
        )
        chunks = splitter.split_documents(docs)
        
        # 3. Adaptive Embedding Generation
        print("\r[3/4] Gerando embeddings...")
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        # 4. Pinecone Operations
        print("\r[4/4] Sincronizando com Pinecone")
        self.manage_pinecone_index("pdfpython", 1024)
        self.upload_vectors("pdfpython", texts, embeddings)

    def generate_embeddings(self, texts):
        batch_size = 4
        embeddings = []
        current_idx = 0
        
        with tqdm(total=len(texts), desc="Gerando embeddings") as pbar:
            while current_idx < len(texts):
                batch = texts[current_idx:current_idx + batch_size]
                
                # GPU-optimized processing
                batch_embeds = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    convert_to_tensor=True,
                    fp16=torch.cuda.is_available()
                )
                
                embeddings.append(batch_embeds.cpu().numpy())
                current_idx += len(batch)
                pbar.update(len(batch))
                
                # Dynamic batch adjustment
                mem_usage = self.gpu_optimizer.get_memory_usage()
                batch_size = self.gpu_optimizer.adjust_batch_size(batch_size, mem_usage)
                
                print(f"\rBatch: {batch_size} | GPU Usage: {mem_usage*100:.1f}%", end="")
        
        return np.concatenate(embeddings, axis=0)

    def manage_pinecone_index(self, index_name, dimension):
        existing_indexes = self.pc.list_indexes().names()
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(2)

    def upload_vectors(self, index_name, texts, embeddings, batch_size=100):
        index = self.pc.Index(index_name)
        total = len(texts)
        
        for i in tqdm(range(0, total, batch_size), desc="Enviando dados"):
            batch = list(zip(texts[i:i+batch_size], embeddings[i:i+batch_size]))
            vectors = [
                (str(j), emb.tolist(), {"text": text})
                for j, (text, emb) in enumerate(batch, start=i)
            ]
            index.upsert(vectors=vectors)

if __name__ == "__main__":
    processor = PDFProcessor()
    processor.process("pdf.pdf")