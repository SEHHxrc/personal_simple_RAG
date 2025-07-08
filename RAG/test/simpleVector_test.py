from SimpleVectorStore import SimpleVectorStore


store = SimpleVectorStore()

# 添加文本
texts = [
    "The cat sat on the mat.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming industries.",
    "FAISS enables efficient similarity search.",
    "Sentence embeddings capture semantic meaning."
]
store.add_texts(texts)

# 测试检索
query = "What tool can do fast similarity search?"
results = store.search(query, k=3)

print(f"Query: {query}")
print("Top 3 similar texts:")
for res in results:
    print("-", res)