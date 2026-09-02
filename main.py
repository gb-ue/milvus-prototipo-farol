#configurar a chamada do milvus
#fazer uma chamada do modulo do clip para processar as imagens e mandar para...
#...o servidor milvus
#fazer o mesmo para o phash (uma próxima etapa pode ser ver como 
#...paralelizar isso)

# a intenção é fazer um ambiente de demonstração pelo que entendi

from module_milvus import *
from module_images import process_data

def main() -> None:

    connections.connect(alias="default", host="localhost", port="19530")

    collection = create_collection(drop=True)
    dir_path = ""

    process_data(dir_path, collection)

    return