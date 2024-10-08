import os
import json
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import SpacyTextSplitter
from tqdm import tqdm
import time
import pandas as pd
import csv

class Document:
    def __init__(self, page_content, lookup_str, metadata, lookup_index):
        self.page_content = page_content
        self.lookup_str = lookup_str
        self.metadata = metadata
        self.lookup_index = lookup_index

    def __repr__(self):
        # 使用repr来确保换行符以文本形式显示
        return f"Document(page_content={repr(self.page_content)}, lookup_str='{self.lookup_str}', metadata={self.metadata}, lookup_index={self.lookup_index})"

def load_documents_from_directory(directory_path):
    documents = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                with open(file_path, mode='r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row_num, row in enumerate(reader, start=1):  # Assuming the first line is the header
                        page_content = f"{row.get('embed', '')}>"
                        # Extract metadata from other columns
                        metadata = {key: value for key, value in row.items() if (key != 'embed')}
                        metadata['row'] = row_num
                        # Create a Document object for each row
                        document = Document(page_content=page_content, lookup_str='', metadata=metadata, lookup_index=0)
                        documents.append(document)
    return documents

# 使用示例 - 现在可以传入一个目录路径
directory_path = 'QA_embed'
unique_docs = load_documents_from_directory(directory_path)

start_time = time.time()
embedding_function = SentenceTransformerEmbeddings(
    model_name='C:\\Users\\WorkPC\\.cache\\huggingface\\hub\\models--BAAI--bge-m3\\snapshots\\5a212480c9a75bb651bcb894978ed409e4c47b82',
    model_kwargs={'device': 'cpu'},
)
end_time = time.time()
print(f"Load Time: {end_time - start_time} seconds")

persist_directory = "BGE_DB4"
print("makeDB")

# 使用批处理创建Chroma数据库
if len(unique_docs) > 1000:
    batch_size = int(len(unique_docs) / 1000)
    batches = [unique_docs[i:i + batch_size] for i in range(0, len(unique_docs), batch_size)]
    for batch in tqdm(batches, desc="Processing batches"):
        db = Chroma.from_documents(batch, embedding_function, persist_directory=persist_directory, collection_metadata={"hnsw:space": "cosine"})
        db.persist()
else:
    db = Chroma.from_documents(unique_docs, embedding_function, persist_directory=persist_directory, collection_metadata={"hnsw:space": "cosine"})
    db.persist()

end_time = time.time()
print(f"embed Time: {end_time - start_time} seconds")
print(db)
