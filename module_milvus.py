from pymilvus import MilvusClient, list_collections, Collection, FieldSchema, CollectionSchema, DataType, SearchResult, milvus_client, connections, utility
from module_images import encode_image_milvus, phash_to_vector
from PIL import Image

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
    
def create_collection(collection_name:str, drop:bool= False):

    if utility.has_collection(collection_name) and drop:
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="phash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=64),
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

    collection = Collection(
        name=collection_name,
        schema=schema,
    )

    collection.create_index(
        field_name="vector",
        index_params=index_params_vector,
    )

    collection.create_index(
        field_name="embedding",
        index_params=index_params_embeddings,
    )

    return collection


# definir função de busca vetorial (por enquanto buscar o top 5 mais semelhantes pode ser útil a nível de demonstração)
# 
# 
#  

def search_image_by_similarity(img_path:str, collection:Collection, top_k:int = 5) -> SearchResult:

    img = Image.open(img_path)

    embedding_from_image = encode_image_milvus(img)

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
    
        # Resultados
    for i, hit in enumerate(result_image_similarity[0]):
           
            img_path_hit = hit.entity.get("path")
            score = hit.score
            threshold = 0.95

            if score >= threshold and img_path_hit != img_path:
                img_path_hit_expr = img_path_hit.replace("\\", "/")

                collection.delete(
                    expr=f'path == "{img_path_hit_expr}"'
                )
                collection.delete(expr=f'path == "{img_path_hit_expr}"')
                collection.flush()
                collection.compact()
                print(f"Imagem {img_path_hit} deletada ({score:.4f}: Imagem duplicada.)")

            print(f"{i + 1}. Score: {score:.4f} | Path: {img_path_hit}")   

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
