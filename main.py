from utils.refine import TextRefiner
import argparse
from utils.misc import make_default_config, load_config
import open_clip
import pickle
import torch
import numpy as np
import os
import os.path as osp
import open3d as o3d
import shutil
import matplotlib.pyplot as plt
 
def process_input(user_input):
    with torch.no_grad():
        text = open_clip_preprocess(user_input).cuda()
        text_feature = open_clip_model.encode_text(text)
        text_feature = text_feature.cpu().numpy()
        return text_feature

def retrieve_3d(text_feature, shape_embeddings, shape_ids, k=5):
    '''
    shape_embeddings: (N, embed_dim)
    text_feature: (1, embed_dim)
    '''

    shutil.rmtree('results', ignore_errors=True)

    similarity = np.dot(shape_embeddings, text_feature.T) / (np.linalg.norm(shape_embeddings, axis=1) * np.linalg.norm(text_feature))
    similarity = [(idx, sim_value) for idx, sim_value in enumerate(similarity.squeeze())]
    similarity.sort(key=lambda x: x[1], reverse=True)

    top_k = similarity[:k]

    os.makedirs('./results', exist_ok=True)
    vis = o3d.visualization.Visualizer()
    img_paths = []
    for idx, _ in top_k:
        target_path = osp.join(config['data_dir'], shape_ids[idx])
        pcd = o3d.io.read_point_cloud(target_path)
        vis.add_geometry(pcd)
        vis.update_renderer()
        vis.capture_screen_image(osp.join('./results', f"{shape_ids[idx]}.png"))
        img_paths.append(osp.join('./results', f"{shape_ids[idx]}.png"))
        vis.clear_geometries()
    vis.destroy_window()

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
    open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', cache_dir="/kaiming-fast-vol/workspace/open_clip_model/")
    open_clip_model = open_clip_model.cuda().eval()

    print('loading Shape Embeddings...')
    
    with open('./modelnet_embed/modelnet.pkl', 'rb') as f:
        shape_embeddings = pickle.load(f)
        shape_ids = np.array(shape_embeddings.keys())
        embeddings = np.array(shape_embeddings.values())
    
    refiner = TextRefiner(access_token=args.access_token)

    while True:
        user_input = input("Enter a user description of shape to retrieve: ")
        k = int(input("Enter the number of shapes to retrieve: "))
        refined_text = refiner.refine(user_input)

        text_feature = process_input(refined_text)
        retrieve_3d(text_feature, shape_embeddings, shape_ids, k=k)
