# Simple RAG Retriever Wrapper
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SimpleVectorStore:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []

    def add_texts(self, texts: list[str]):
        """
        快速添加文本，不写入文件
        :param texts: 文本内容
        :return:
        """
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        embeddings = np.array(embeddings).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.documents.extend(texts)

    def search(self, query, k=5) -> list[str] | None:
        query_vec = self.model.encode([query], convert_to_tensor=False).astype("float32")
        if self.is_empty():
            return None
        distances, indices = self.index.search(query_vec, k)
        return [self.documents[i] for i in indices[0]]

    def is_empty(self) -> bool:
        if self.index is None or len(self.documents) <= 0:
            return True
        return False
