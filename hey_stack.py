import os
import uuid
import json
import requests
from haystack import Pipeline, Document, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever
from typing import List

from embed_api import get_embed

#  MilvusDocumentStore 설정
document_store = MilvusDocumentStore(
    collection_name="information_db",
    connection_args={"uri": "http://" + os.getenv('MILVUS', 'localhost') + ":19530"},
    index_params={
        "index_type": "GPU_CAGRA",
        "metric_type": "L2",
        "params": {
            "intermediate_graph_degree": 64,
            "graph_degree": 32,
            "build_algo": "NN_DESCENT",
            "cache_dataset_on_device": "false"
        }
    },
    consistency_level="Session",
    drop_old=False,  # 기존 컬렉션을 삭제하지 않음
    primary_field="id",
    text_field="content",
    vector_field="embed",
)

# 4. 데이터 삽입 함수 정의
def insert_data(summarized, urls):
    if len(summarized) != len(urls):
        raise ValueError("summarized와 urls의 길이가 다릅니다.")

    print("Before Number of documents:", document_store.count_documents())
    documents = [
        Document(content=summarized[i], meta={'source_url': urls[i]}, id=str(uuid.uuid4()))
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

# 5. 샘플 데이터 삽입 (테스트용)
if __name__ == '__main__':
    sample_summaries = ["이것은 첫 번째 문서입니다.", "이것은 두 번째 문서입니다."]
    sample_urls = ["https://example.com/1", "https://example.com/2"]
    insert_data(sample_summaries, sample_urls)