from sentence_transformers import SentenceTransformer
import faiss
import os
import json

class RAGRetriever:
    def __init__(self, kb_path="data/knowledge_base.txt", index_dir="data/faiss_index"):
        self.kb_path = kb_path
        self.index_dir = index_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._load_or_build_index()

    def _load_or_build_index(self):
        if os.path.exists(self.index_dir):
            self.index = faiss.read_index(os.path.join(self.index_dir, "kb.index"))
            with open(os.path.join(self.index_dir, "corpus.json")) as f:
                self.corpus = json.load(f)
        else:
            with open(self.kb_path, 'r') as f:
                self.corpus = [line.strip() for line in f if line.strip()]
            embeddings = self.model.encode(self.corpus, convert_to_numpy=True)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            os.makedirs(self.index_dir, exist_ok=True)
            faiss.write_index(self.index, os.path.join(self.index_dir, "kb.index"))
            with open(os.path.join(self.index_dir, "corpus.json"), 'w') as f:
                json.dump(self.corpus, f)

    def retrieve(self, query, k=3):
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(query_embedding, k)
        return [self.corpus[i] for i in indices[0]]
