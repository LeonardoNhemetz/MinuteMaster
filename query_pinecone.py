from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import torch
import os

load_dotenv()

class VectorQueryEngine:
    def __init__(self, index_name="pdfpython"):
        self.model = SentenceTransformer(
            "billatsectorflow/stella_en_1.5B_v5",
            device="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)
    
    def semantic_search(self, query: str, top_k: int = 3):
        # Gerar embedding da consulta
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            fp16=torch.cuda.is_available()
        ).cpu().numpy().tolist()
        
        # Executar consulta no Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        
        return self._format_results(results)

    def _format_results(self, results):
        formatted = []
        for match in results['matches']:
            formatted.append({
                'score': match['score'],
                'text': match['metadata']['text'],
                'id': match['id']
            })
        return formatted

if __name__ == "__main__":
    query_engine = VectorQueryEngine()
    question = "What the document talk about?"
    
    print(f"🔍 Searching for: '{question}'")
    results = query_engine.semantic_search(question, top_k=5)
    
    print("\nTop Relevant Results:")
    for i, result in enumerate(results, 1):
        print(f"\n📄 Result #{i} (Score: {result['score']:.4f})")
        print(f"ID: {result['id']}")
        print(f"Content:\n{result['text']}\n{'-'*50}")