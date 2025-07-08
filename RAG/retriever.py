# rag/retriever.py
import faiss
import numpy as np
from langchain.text_splitter import CharacterTextSplitter

class FaissRetriever:
    """
    根据预编码词向量，选取匹配度高的文档
    """
    def __init__(self, embedder, index_path="knowledge_base/index.faiss", corpus_path="knowledge_base/corpus.txt"):
        self.embedder = embedder
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.docs = []
        self.index = None

    def build(self, chunk_size=300, chunk_overlap=50):
        """
        根据文件内容构建索引
        :param chunk_size: 文档分块长度
        :param chunk_overlap:
        :return:
        """
        try:
            text = open(self.corpus_path, encoding="utf-8").read()
        except Exception as e:
            print(e)
            exit(1)
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if self.is_empty():
            print('no file in dir.')
            return
        self.docs = splitter.split_text(text)
        vectors = self.embedder.encode(self.docs)
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(np.array(vectors))
        faiss.write_index(self.index, self.index_path)

    def load(self):
        """
        根据索引导入文档内容
        """
        self.index = faiss.read_index(self.index_path)
        try:
            text = open(self.corpus_path, encoding="utf-8").read()
        except Exception as e:
            print(e)
            exit(1)
        splitter = CharacterTextSplitter()
        self.docs = splitter.split_text(text)

    def retrieve(self, query, top_k=3) -> list[dict[str, float]]:
        """
        返回top_k个文件内容和相似度，按相似度从高到低排序（不一定是得分最高）
        :param query: 查询语句的词向量
        :param top_k: 返回最相似/相关的top_k个结果
        :return: "文档：相似度"序列
        """
        if self.index is None:
            self.load()
        query_vec = self.embedder.encode([query])
        D, I = self.index.search(np.array(query_vec), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({"text": self.docs[idx], "score": float(score)})
        return results

    def is_empty(self) -> bool:
        if len(self.docs) <= 0:
            return True
        return False
