from embedder import LocalEmbedder


embedders = LocalEmbedder()
result = embedders.encode('')

result1 = embedders.encode('')

from numpy import dot
from numpy.linalg import norm
similarity = dot(result, result1) / (norm(result) * norm(result1))
print(f"相似度：{similarity:.4f}")
