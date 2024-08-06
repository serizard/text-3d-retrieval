from utils.refine import TextRefiner
import argparse
from utils.renderer import Renderer
from utils.misc import make_default_config, load_config
import open_clip
import pickle
import numpy as np
import os
import os.path as osp
import shutil
import matplotlib.pyplot as plt
import open3d as o3d
from utils.data import process_input




def rerank_candidates(candidates, shape_ids, text_feature, keyword_embeddings, k):
    reranked_candidates = []
    for idx, _ in candidates:
        src_path = shape_ids[idx]
        keyword = [key for key in keyword_embeddings.keys() if key in src_path][0]
        keyword_feature = keyword_embeddings[keyword]

        reranked_similarity = np.dot(keyword_feature, text_feature.T)
        reranked_candidates.append((idx, reranked_similarity[0])) 

    reranked_candidates.sort(key=lambda x: x[1], reverse=True)

    return reranked_candidates[:k]


def retrieve_3d(text_feature, shape_embeddings, shape_ids, keyword_embeddings, k=5, rerank=True):
    '''
    shape_embeddings: (N, embed_dim)
    text_feature: (1, embed_dim)
    keywords: list of lists, each inner list contains keywords for corresponding shape_id
    keyword_embeddings: dictionary {keyword: embedding}
    '''

    shutil.rmtree('results', ignore_errors=True)
    os.makedirs('./results', exist_ok=True)

    if k > len(shape_ids):
        raise ValueError(f"Number of shapes to retrieve should be less than {len(shape_ids)}")
    
    candidate_num = min(k*3, len(shape_ids))

    similarity = np.dot(shape_embeddings, text_feature.T) # (N, 1)
    indexed_similarity = [(idx, sim_value) for idx, sim_value in enumerate(similarity.squeeze())]
    indexed_similarity.sort(key=lambda x: x[1], reverse=True)

    candidates = indexed_similarity[:candidate_num]

    if rerank:
        top_k = rerank_candidates(candidates, shape_ids, text_feature, keyword_embeddings, k)
    else:
        top_k = candidates[:k]

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
    parser.add_argument('--rerank', type=bool, default=True, help='Rerank the candidates by text similarity')
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
        modelnet_embeddings = pickle.load(f)
        shape_ids = np.array(list(modelnet_embeddings.keys())) # (N,)
        shape_embeddings = np.array(list(modelnet_embeddings.values())) # (N, embed_dim)

    with open('./modelnet_embed/modelnet_name.pkl', 'rb') as f:
        keyword_embeddings = pickle.load(f)
    
    refiner = TextRefiner()
    renderer = Renderer(config.rendering_width, config.rendering_height)

    user_input = input("Enter a user description of shape to retrieve: ")
    k = int(input("Enter the number of shapes to retrieve: "))
    refined_text = [refiner.refine(user_input)]
    print(f'user input: {user_input}, refined_text: {refined_text[0]}')

    text_feature = process_input(refined_text, open_clip_model, device)[0] # (1, embed_dim,)
    off_paths = retrieve_3d(text_feature, shape_embeddings, shape_ids, keyword_embeddings, k=k, rerank=args.rerank)

    for i, off_path in enumerate(off_paths):
        obj_file = o3d.io.read_triangle_mesh(off_path)
        o3d.visualization.draw_geometries([obj_file])


