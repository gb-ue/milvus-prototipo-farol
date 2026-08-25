import clip, torch, os
from PIL import Image
from pymilvus import Collection

#atualmente temos apenas as instanciaçoes de algumas funções para trabalhar com o clip para imagens
#ele codifica a imagem e faz a indexação dela (chama o encoding)
#aao que tudo indica a funcionalidade dele termina aqui, depois buscar finalizar as funções do milvus

model, preprocess = clip.load("ViT-B/32")
model.eval()

def encode_image(image_path: str):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image)

    image_features /= image_features.norm(
        dim=-1, keepdim=True
    )

    return image_features.squeeze().tolist()

def encode_text(text: str):
    text_tokens = clip.tokenize(text)
    with torch.no_grad():
        text_features = model.encode(text_tokens)

    text_features /= text_features.norm(
        dim=-1, keepdim=True
    )

    return text_features.squeeze().tolist()

def index_image(folder_path:str, collection:Collection):
    embeddings = []
    paths = []

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".jfif")):
            path = os.path.join(folder_path,file)
            emb = encode_image(path)
            embeddings.append(emb.tolist())
            paths.append(path)
    if embeddings:
        collection.insert([embeddings, paths])
        collection.flush()
    return True

print(encode_text("boa noite"))