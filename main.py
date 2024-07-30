from utils.refine import TextRefiner
import argparse
from configparser import ConfigParser
from utils.misc import make_default_config, load_config, dump_config
from download_dataset import download_modelnet40
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
    '''
    user_input 받아서 사전에 정의된 clip 모델 바탕으로 text feature로 변환하기
    
    by 도형
    '''

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
    
    plt.title("Retrieved 3D objects")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process user input')
    parser.add_argument('--access_token', type=str, help='Hugging Face access token', required=False)
    parser.add_argument('--init_config', type=bool, help='Initialize default configuration')
    parser.add_argument('--download', type=bool, default=False, help='Download ModelNet40 dataset')
    parser.add_argument('--download_dir', type=str, default='./data', help='Directory to download 3D objects')
    args = parser.parse_args()

    if args.init_config or not osp.exists('./utils/configs/config.ini'):
        make_default_config()
    
    config = load_config()

    if args.download:
        dataset_path = download_modelnet40(args)

    print("loading OpenCLIP model...")
    open_clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', cache_dir="/kaiming-fast-vol/workspace/open_clip_model/")
    if torch.cuda.is_available():
        open_clip_model = open_clip_model.cuda()
        open_clip_model = torch.nn.DataParallel(open_clip_model)
        open_clip_model = open_clip_model.half()

    open_clip_model.eval()

    print('loading Shape Embeddings...')
    
    with open('./modelnet_embed/modelnet.pkl', 'rb') as f:
        shape_embeddings = pickle.load(f)
        shape_ids = np.array(shape_embeddings.keys())
        embeddings = np.array(shape_embeddings.values())

    
    refiner = TextRefiner(access_token=args.access_token)

    while True:
        user_input = input("Enter a user description of shape to retrieve: ")
        structured_text = refiner.refine(user_input)

        text_feature = process_input(structured_text)
        retrieve_3d(text_feature, shape_embeddings, shape_ids)
