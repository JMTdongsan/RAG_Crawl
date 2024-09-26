import os

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, db

# Milvus 서버에 연결
conn = connections.connect(host=os.getenv('MILVUS'), port=19530)

# 컬렉션 이름 정의
collection_name = "information_db"

# 필드 스키마 정의
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="embed", dtype=DataType.FLOAT_VECTOR, dim=1024)
]

# 컬렉션 스키마 생성
schema = CollectionSchema(fields, description="Information database")

# 컬렉션 생성
collection = Collection(name=collection_name, schema=schema)

# GPU_BRUTE_FORCE 인덱스 파라미터 설정
index_params = {
    'index_type': 'GPU_BRUTE_FORCE',
    'metric_type': 'L2',
    'params': {}
}

# 인덱스 생성
collection.create_index(field_name="embed", index_params=index_params)

print(f"Collection '{collection_name}' created successfully with GPU_BRUTE_FORCE index.")

# 컬렉션 로드 (검색을 위해 필요)
collection.load()