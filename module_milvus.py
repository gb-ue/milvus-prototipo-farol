from pymilvus import MilvusClient, connections, list_collections, Collection, FieldSchema, CollectionSchema, DataType
from module_images import encode_image_milvus

client = MilvusClient("milvus_demo.db") 

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

def create_index(collection:Collection):
    collection.create_index("embedding"), {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 1024}
    }
    

def start_collection(collection_name:str):
    
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)

    if collection_name == "demo_collection":
        client.create_collection(
            collection_name=collection_name,
            dimension=768,
        )

    elif collection_name == "image_collection":
        client.create_collection(
            collection_name=collection_name,
            dimension=512,
            auto_id=True,
            enable_dynamic_field=True,
        )

def create_embedding_collection():

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), 
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500)
        ]
    schema = CollectionSchema(fields, description="Image Similarity Search")
    
    if "image_embeddings" in list_collections():
        Collection("image_embeddings").drop()
    
    client.create_collection(
        collection_name="image_embeddings",
        schema=schema
        )

# definir função de busca vetorial (por enquanto buscar o top 5 mais semelhantes pode ser útil a nível de demonstração)
# 
# 
#  

def search_similar_image(img_path:str, collection:Collection, top_k:int = 10):
    emb = encode_image_milvus(img_path)
    result = collection.search(
        data=[emb],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["path"]
    )
    return result[0]

def start_connection():
    connections.connect(alias="default", host="localhost", port="19530")