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

    collection = create_collection("image_data_farol", drop=True)
    #exemplo hardcodado
    dir_path = "C:\\Users\\user\\Documents\\vscode-stuff\\Trabalho\\OCR\\PIDF"

    process_data(dir_path, collection)

    print("ENTIDADES NO MILVUS:", collection.num_entities)

    collection.load()

    search_image_by_similarity("c:\\Users\\user\\Documents\\vscode-stuff\\Trabalho\\OCR\\PIDF\\0008a8c8129cc1d79a25299904f84931.jpg", collection)

main()