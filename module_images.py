from PIL import Image
from pymilvus import Collection
import imagehash, clip, torch, os

# esse módulo está em produção para unificar as funões dos módulos clip e phash, 
# visto suas semelhanças nas funções principais e a necessidade de fazer com que 
# o processamento dos embeddings e do phash sejam unificados
# verificar a inicialização do banco e das tabelas depois ver como elas estão com um exemplo simples
# isso deve concluir tudo de relevante para fazer o experimento 

model, preprocess = clip.load("ViT-B/32")
model.eval()

#gera o embedding da imagem a partir de um caminho processado pelo process data (abaixo)
def encode_image_milvus(img: Image): 

    img = img.convert("RGB")

    preprocessed_clip_img = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        image_features = model.encode_image(preprocessed_clip_img)
    
    image_features /= image_features.norm(
        dim=-1, keepdim=True
    )

    return image_features.squeeze().tolist()

# gera um vector do phash criado dentro do process data
def phash_to_vector(phash_str: str):
    binary = bin(int(phash_str, 16))[2:].zfill(64)

    return [float(bit) for bit in binary]

#recebe uma entrada de uma pasta e processa as imagens que estão nela compostas
def process_data(folder_path:str, collection:Collection):

    paths = []
    phashes = []
    vectors = []
    embeddings = []

    for file in os.listdir(folder_path):
    
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".jfif")):
            filepath = os.path.join(folder_path, file)
            print("\nProcessando:", filepath)

            try:
                img = Image.open(filepath)

                emb = encode_image_milvus(img)
                phash = str(imagehash.phash(img))
                vector = phash_to_vector(phash)

                print("Embedding:", len(emb))
                print("pHash:", phash)
                print("Vector:", len(vector))

                paths.append(filepath)
                phashes.append(phash)
                vectors.append(vector)
                embeddings.append(emb)
                
            except Exception as e:
                print(f"A imagem {filepath} possui um erro inexperado:\n{e}")

    if paths:
        collection.insert([paths, embeddings, phashes, vectors])
        collection.flush()

    else:
        print("\nNENHUMA IMAGEM FOI PROCESSADA!")