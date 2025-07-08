# rag/embedder.py
from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    """
    词向量编码，交给其他模块匹配相似度
    """
    def __init__(self, model_name_or_path="BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(model_name_or_path)

    def encode(self, texts, normalize=True):
        return self.model.encode(texts, normalize_embeddings=normalize, convert_to_numpy=True)
