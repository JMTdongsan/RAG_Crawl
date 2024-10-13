import uuid
import json
import requests
from haystack import Pipeline, Document, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever
from typing import List

from config import MILVUS
from embed_api import get_embed

#  MilvusDocumentStore 설정
document_store = MilvusDocumentStore(
    collection_name="information_db",
    connection_args={"uri": "http://" + MILVUS + ":19530"},
    consistency_level="Session",
    drop_old=False,
    primary_field="id",
    text_field="content",
    vector_field="embed",
    index_params={
        "index_type": "GPU_CAGRA",
        "metric_type": "L2",
        "params": {
            "intermediate_graph_degree": 64,
            "graph_degree": 32,
            "build_algo": "NN_DESCENT",
            "cache_dataset_on_device": "false"
        }
    }
)

# 데이터 삽입 함수 정의
def insert_data(summarized: List[str], urls: List[str]):
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
    sample_summaries = ["정비사업은 일종의 소형 재개발 사업으로서, 노후·불량건축물이 밀집한 가로구역에서 종전의 가로를 유지하면서 소규모로 주거환경을 개선하기 위하여 시행하는, 소규모주택정비사업의 하나이다. 2018년 「빈집 및 소규모주택 정비에 관한 특례법」 개정 시 주거환경관리사업과 함께 정비사업 유형의 하나로 도입되었으며, 기존 저층주거지의 도시조직과 가로망을 유지하며 주거환경을 개선하기 위하여 시행하는 소규모 사업이다.",
                        "정비사업 정보몽땅은 클린업시스템, 사업비 및 분담금 추정프로그램, e-조합시스템의 개별 운영 및 사용자관리, 로그인, 정보공개 등 기능 중복으로 사용자 불편 및 혼란에 따른 민원을 해소하고자 이를 통합·일원화하여 정비사업 투명성 강화 및 이용편의성을 개선한 종합정보관리시스템입니다."]
    sample_urls = ["https://example.com/1", "https://example.com/2"]
    insert_data(sample_summaries, sample_urls)