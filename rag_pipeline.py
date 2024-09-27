import os

from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from milvus_haystack import MilvusEmbeddingRetriever, MilvusDocumentStore

from embed_api import CustomTextEmbedder

document_store = MilvusDocumentStore(
    collection_name="information_db",
    connection_args={"uri": "http://"+os.getenv('MILVUS')+":19530"},
)

prompt_template = """think english, and reply in korean, think step by step
                        Answer the following query based on the provided context. If the context does
                     not include an answer, reply with 'I don't know'. If you want to question, just do it\n
                     Query: {{query}}
                     Documents:
                     {% for doc in documents %}
                        {{ doc.content }}
                     {% endfor %}
                     Answer: 
                  """

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", CustomTextEmbedder())
rag_pipeline.add_component("retriever", MilvusEmbeddingRetriever(document_store=document_store, top_k=3))
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
rag_pipeline.add_component("generator", OpenAIGenerator(generation_kwargs={"temperature": 0}))
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "generator")


if __name__ == '__main__':
    question = "오늘의 날씨는?"
    results = rag_pipeline.run(
        {
            "text_embedder": {"text": question},
            "prompt_builder": {"query": question},
        }
    )
    print('RAG answer:', results["generator"]["replies"][0])