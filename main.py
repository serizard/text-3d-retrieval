from utils.refine import TextRefiner
import argparse
from utils.renderer import Renderer
from utils.misc import make_default_config, load_config
import open_clip
import pickle
import torch
import numpy as np
import os
import os.path as osp
import shutil
import matplotlib.pyplot as plt
import open3d as o3d

 
@torch.no_grad()
def process_input(user_input, open_clip_model, device):
    with torch.no_grad():
        text = open_clip.tokenizer.tokenize(user_input).to(device)
        return open_clip_model.encode_text(text).cpu().numpy()


def retrieve_3d(text_feature, shape_embeddings, shape_ids, config, k=5):
    '''
    shape_embeddings: (N, embed_dim)
    text_feature: (1, embed_dim)
    '''

    shutil.rmtree('results', ignore_errors=True)
    os.makedirs('./results', exist_ok=True)

    similarity = np.dot(shape_embeddings, text_feature.T) # (N, 1)
    indexed_similarity = [(idx, sim_value) for idx, sim_value in enumerate(similarity.squeeze())]
    indexed_similarity.sort(key=lambda x: x[1], reverse=True)

    top_k = indexed_similarity[:k]

    off_paths = []
    for i, (idx, _) in enumerate(top_k):
        src_path = shape_ids[idx]
        dest_path = osp.join('./results', f'Rank_{i+1}_{osp.basename(src_path)}')
        shutil.copy(src_path, dest_path)
        off_paths.append(src_path)
    
    return off_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process user input')
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
    except:
        print("Error occurred. Running on CPU.")
        open_clip_model = open_clip_model.cpu().eval()
        device = 'cpu'

    print('loading Shape Embeddings...')
    
    with open('./modelnet_embed/modelnet.pkl', 'rb') as f:
        shape_embeddings = pickle.load(f)
        shape_ids = np.array(list(shape_embeddings.keys())) # (N,)
        embeddings = np.array(list(shape_embeddings.values())) # (N, embed_dim)
    
    refiner = TextRefiner()
    renderer = Renderer(config.rendering_width, config.rendering_height)

    user_input = input("Enter a user description of shape to retrieve: ")
    k = int(input("Enter the number of shapes to retrieve: "))
    refined_text = [refiner.refine(user_input)]
    print(f'user input: {user_input}, refined_text: {refined_text[0]}')

    text_feature = process_input(refined_text, open_clip_model, device)[0] # (1, embed_dim,)
    off_paths = retrieve_3d(text_feature, embeddings, shape_ids, config, k=k)

    _, axs = plt.subplots(1, k, figsize=(k*5, 5))

    for i, off_path in enumerate(off_paths):
        obj_file = o3d.io.read_triangle_mesh(off_path)
        rgb = renderer.render(obj_file)
        axs[i].imshow(rgb)
        axs[i].axis('off')
        axs[i].set_title(f'Rank {i+1}')
    
    plt.savefig('./results/visualization.png')
    renderer.close()
        


