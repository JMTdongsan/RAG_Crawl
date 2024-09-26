from haystack import Document
from milvus_haystack import MilvusDocumentStore

document_store = MilvusDocumentStore(
    connection_args={"uri": "http://lee5j.iptime.org:19530"},  # Milvus Lite
    # connection_args={"uri": "http://localhost:19530"},  # Milvus standalone docker service.
    drop_old=True,
)
documents = [Document(
    content="A Foo Document",
    meta={"article": "{url}", "chapter": "intro"},
    embedding=[-10.0] * 128,
)]
document_store.write_documents(documents)
print(document_store.count_documents())  # 1
