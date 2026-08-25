from pymilvus import MilvusClient, connections, list_collections, Collection, FieldSchema, CollectionSchema, DataType

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

def create_embedding_collection(): # talvez adicionar collection_name:str

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

def start_connection():
    connections.connect(alias="default", host="localhost", port="19530")