import os
from langchain.document_loaders import TextLoader, PDFPlumberLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter
from embedder import LocalEmbedder
import faiss
import numpy as np

def load_documents_from_folder(folder_path):
    loaders = {
        ".txt": TextLoader,
        ".pdf": PDFPlumberLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    documents = []
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[-1].lower()
        if ext in loaders:
            path = os.path.join(folder_path, filename)
            try:
                loader = loaders[ext](path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
    return documents

def ingest_documents(
    folder_path="knowledge_base",
    index_path="knowledge_base/index.faiss",
    corpus_path="knowledge_base/corpus.txt",
    chunk_size=300,
    chunk_overlap=50,
    embed_model="BAAI/bge-small-zh-v1.5"
):
    print("正在加载文档...")
    documents = load_documents_from_folder(folder_path)
    texts = [doc.page_content for doc in documents]

    print(f"正在切分为 chunk（大小 {chunk_size}，重叠 {chunk_overlap}）...")
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents(texts)
    chunk_texts = [chunk.page_content for chunk in chunks]

    print("正在编码向量...")
    embedder = LocalEmbedder(embed_model)
    vectors = embedder.encode(chunk_texts)

    print(f"正在构建索引并保存至 {index_path}...")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    faiss.write_index(index, index_path)

    print(f"正在保存原始分块文本到 {corpus_path}...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in chunk_texts:
            f.write(text.strip() + "\n")

    print("ingest 完成")

if __name__ == "__main__":
    ingest_documents()
