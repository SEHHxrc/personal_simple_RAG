from ingest import ingest_documents

ingest_documents('knowledge_base',
                 index_path='knowledge_base/index.faiss',
                 corpus_path='knowledge_base/corpus.txt',
                 embed_model='')
