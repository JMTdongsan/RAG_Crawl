
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from milvus_haystack import MilvusEmbeddingRetriever, MilvusDocumentStore

from embed_api import CustomTextEmbedder
from hey_stack import document_store
from send_llm import CustomGenerator

prompt_template = """think english, and reply in korean
                     Answer the following query based on the provided context. If the context does not include an answer, 
                     If you can't answer with Documents, you can use searching tool\n
                     Query: {{query}}
                     Documents:
                     {% for doc in documents %}
                        {{ doc.content }}
                     {% endfor %}
                     Answer: 
                  """

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", CustomTextEmbedder())
rag_pipeline.add_component("retriever", MilvusEmbeddingRetriever(document_store=document_store, top_k=5))
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
rag_pipeline.add_component("generator", CustomGenerator())
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "generator")


if __name__ == '__main__':
    question = "what is REITs?"
    results = rag_pipeline.run(
        {
            "text_embedder": {"text": question},
            "prompt_builder": {"query": question},
        }
    )
    print('RAG answer:', results["generator"]["replies"][0])