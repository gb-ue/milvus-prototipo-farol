import imagehash, os
from PIL import Image
from pymilvus import Collection

def phash_to_vector(phash_str):
    binary = bin(int(phash_str, 16))[2:].zfill(64)

    return [float(bit) for bit in binary]

def hash_image_folder(folder_path:str, collection:Collection):
    paths = []
    phashes = []
    vectors = []

    for file in os.listdir(folder_path):

        filepath = os.path.join(folder_path, file)
        if not os.path.isfile(filepath):
            continue

        try:
            img = Image.open(filepath)
            phash = str(imagehash.phash(img))
            vector = phash_to_vector(phash)

            paths.append(filepath)
            phashes.append(phash)
            vectors.append(vector)

        except Exception as e:
            print(e)

    if paths:
        data = [
            paths,
            phashes,
            vectors
        ]

        collection.insert(data)
        collection.flush()