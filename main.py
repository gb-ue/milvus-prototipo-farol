#configurar a chamada do milvus
#fazer uma chamada do modulo do clip para processar as imagens e mandar para...
#...o servidor milvus
#fazer o mesmo para o phash (uma próxima etapa pode ser ver como 
#...paralelizar isso)

# a intenção é fazer um ambiente de demonstração pelo que entendi

from pymilvus import MilvusClient, connections, list_collections, Collection, FieldSchema, CollectionSchema, DataType, SearchResult

def main() -> None:

    connections.connect(alias="default", host="localhost", port="19530")

    collection = create_

    return