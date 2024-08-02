from utils.refine import TextRefiner
import argparse
from utils.misc import make_default_config, load_config
import open_clip
import pickle
import torch
import numpy as np
import os
import os.path as osp
import shutil
import matplotlib.pyplot as plt
import trimesh
 
@torch.no_grad()
def process_input(user_input, open_clip_model, device):
    with torch.no_grad():
        text = open_clip.tokenizer.tokenize(user_input).to(device)
        return open_clip_model.encode_text(text).cpu().numpy()


def make_image(mesh, output_path):
    scene = mesh.scene()
    png = scene.save_image(resolution=[800, 800], visible=True)

    with open(output_path, 'wb') as f:
        f.write(png)


def retrieve_3d(text_feature, shape_embeddings, shape_ids, k=5):
    '''
    shape_embeddings: (N, embed_dim)
    text_feature: (1, embed_dim)
    '''

    os.makedirs('./results', exist_ok=True)
    shutil.rmtree('results', ignore_errors=True)

    similarity = np.dot(shape_embeddings, text_feature.T) # (N, 1)
    indexed_similarity = [(idx, sim_value) for idx, sim_value in enumerate(similarity.squeeze())]
    indexed_similarity.sort(key=lambda x: x[1], reverse=True)

    top_k = indexed_similarity[:k]

    img_paths = []
    for idx, _ in top_k:
        target_path = shape_ids[idx]
        mesh = trimesh.load(target_path, file_type='off')
        result_save_path = osp.join('./results', f"{osp.basename(target_path).split('.')[0]}.png")
        scene = mesh.scene()
        png = scene.save_image(resolution=[800, 800], visible=True)

        with open(result_save_path, 'wb') as f:
            f.write(png)
            
        img_paths.append(result_save_path)

    _, axes = plt.subplots(1, k, figsize=(15, 5))
    for i, img_path in enumerate(img_paths):
        axes[i].imshow(plt.imread(img_path))
        axes[i].axis("off")
        axes[i].set_title(f"Rank {i+1}")
    
    plt.title("Retrieved 3D objects")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process user input')
    parser.add_argument('--access_token', type=str, help='Hugging Face access token')
    parser.add_argument('--init_config', type=bool, help='Initialize default configuration')
    args = parser.parse_args()

    if args.init_config or not osp.exists('./utils/configs/config.json'):
        make_default_config()
    
    config = load_config()

    print("loading OpenCLIP model...")
    os.makedirs('./clip_cache', exist_ok=True)

    open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', cache_dir='./clip_cache')
    try:
        open_clip_model = open_clip_model.cuda().eval()
        device = 'cuda'
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Running on CPU.")
        open_clip_model = open_clip_model.cpu().eval()
        device = 'cpu'

    print('loading Shape Embeddings...')
    
    with open('./modelnet_embed/modelnet.pkl', 'rb') as f:
        shape_embeddings = pickle.load(f)
        shape_ids = np.array(list(shape_embeddings.keys())) # (N,)
        embeddings = np.array(list(shape_embeddings.values())) # (N, embed_dim)
    
    # refiner = TextRefiner(access_token=args.access_token)

    while True:
        user_input = input("Enter a user description of shape to retrieve: ")
        k = int(input("Enter the number of shapes to retrieve: "))
        # refined_text = refiner.refine(user_input)
        refined_text = [user_input]

        text_feature = process_input(refined_text, open_clip_model, device)[0] # (1, embed_dim,)
        retrieve_3d(text_feature, embeddings, shape_ids, k=k)
