from embedder import LocalEmbedder
from retriever import FaissRetriever


embedder = LocalEmbedder()
retriever = FaissRetriever(embedder, index_path='knowledge_base/index.faiss', corpus_path='knowledge_base/corpus.txt')

retriever.load()

query = ''

result = retriever.retrieve(query)
for i, doc in enumerate(result):
    print(f"Result {i+1}: {doc['text']} (Score: {doc['score']:.4f})")
