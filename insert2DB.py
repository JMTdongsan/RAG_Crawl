import uuid
from haystack import  Document
from typing import List

from vector_db import document_store
from embed_api import get_embed



# 데이터 삽입 함수 정의
def insert_data(summarized: List[str], urls: List[str]):
    if len(summarized) != len(urls):
        raise ValueError("summarized와 urls의 길이가 다릅니다.")

    print("Before Number of documents:", document_store.count_documents())
    documents = [
        Document(content=summarized[i], meta={'source_url': urls[i]})#, id=str(uuid.uuid4())
        for i in range(len(summarized))
    ]

    # 임베딩 벡터를 배치로 생성
    contents = [doc.content for doc in documents]
    embeddings = get_embed(contents)  # get_embed 함수가 리스트를 반환함

    for doc, embedding in zip(documents, embeddings):
        doc.embedding = embedding

    # 문서 저장
    document_store.write_documents(documents)
    print("After Number of documents:", document_store.count_documents())


