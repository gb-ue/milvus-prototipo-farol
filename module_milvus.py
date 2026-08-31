from pymilvus import MilvusClient, list_collections, Collection, FieldSchema, CollectionSchema, DataType, SearchResult, milvus_client, connections, utility
from module_images import encode_image_milvus, phash_to_vector

#client = MilvusClient("milvus_demo.db") 

#"image_collection"
#"demo_collection"

#progresso:
#atualmente apenas chamando algumas funções básicas para chamar os processos básicos do client de criar coleções e começar conexão
#depois fazer as funções para busca por similaridade e alguns afins a serem discutidos com o pessoal
#
#
#
#
#
#

#restirar para colocar na main*
def create_index(collection:Collection):
    collection.create_index("embedding"), {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 1024}
    }
    
def create_collection(collection_name:str, drop: bool):
    
    if milvus_client.has_collection(collection_name=collection_name):
        milvus_client.drop_collection(collection_name=collection_name)

    if utility.has_collection(collection_name):
        milvus_client.create_collection(
            collection_name,
        )

    elif "image_embeddings" in list_collections():
        Collection("image_embeddings").drop()

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="phash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="vectore", dtype=DataType.FLOAT_VECTOR, max_length=64),
        ]
    schema = CollectionSchema(fields, description="Image Similarity Search")

    index_params_vector = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {
            "nlist": 128,
        },
    }

    index_params_embeddings = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {
            "nlist": 512,
        },
    }
    
    milvus_client.create_collection(
        collection_name="image_embeddings",
        schema=schema
        )

    milvus_client.create_index(
        field_name="vector",
        index_params=index_params_vector
    )

    milvus_client.create_index(
        field_name="embeddings",
        index_params=index_params_embeddings
    )

#restirar para colocar na main*
def create_embedding_collection():

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), 
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500)
        ]
    schema = CollectionSchema(fields, description="Image Similarity Search")
    
    if "image_embeddings" in list_collections():
        Collection("image_embeddings").drop()
    
    milvus_client.create_collection(
        collection_name="image_embeddings",
        schema=schema
        )

# definir função de busca vetorial (por enquanto buscar o top 5 mais semelhantes pode ser útil a nível de demonstração)
# 
# 
#  

def search_image_by_similarity(img_path:str, collection:Collection, top_k:int = 5) -> SearchResult:

    embedding_from_image = encode_image_milvus(img_path)

    result_image_similarity = collection.search(
        data=[embedding_from_image],
        anns_field="embedding",
        param={
            "metric_type": "COSINE", 
            "params": {
                "nprobe": 10
            },
        },
        limit=top_k,
        output_fields=[
            "path",
            "embedding"
        ],
    )
    return result_image_similarity

def search_image_by_phash(phash:str, collection:Collection, top_k:int = 5) -> SearchResult:
    #referenciado do github do davi:

    vector_from_phash = phash_to_vector(phash)

    result_phash_search = collection.search(
        data=[vector_from_phash],
        anns_field="vector",
        param={
            "metric_type": "L2",
            "params": {
                "nprobe":10
            },
        },
        limit=top_k,
        output_fields=[
            "image_path",
            "phash",
        ],
    )

    return result_phash_search

#restirar para colocar na main*
def start_connection():
    connections.connect(alias="default", host="localhost", port="19530")