import os
import urllib.request
import zipfile
import shutil
import argparse
import open3d as o3d
from utils.misc import load_config
from glob import glob
from tqdm import tqdm
import os.path as osp
import numpy as np
from utils.data import normalize_pc
from collections import OrderedDict
import re
import torch
import model.models as models
import pickle
from huggingface_hub import hf_hub_download
from utils.misc import dump_config

def move_off_files(src_dir, dest_dir):
    """Move .off files from the source directory to the destination directory."""
    if not osp.exists(dest_dir):
        os.makedirs(dest_dir)
    
    off_files = glob(osp.join(src_dir, '**/*.off'), recursive=True)
    for file_path in off_files:
        dest_file_path = osp.join(dest_dir, osp.basename(file_path))
        shutil.move(file_path, dest_file_path)

def download_modelnet10(data_dir):
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        progress = downloaded / total_size * 100
        print(f"\rProgress: {progress:.2f}%", end="")

    os.makedirs(data_dir, exist_ok=True)
    """Download and extract the ModelNet10 dataset."""
    url = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
    zip_path = osp.join(data_dir, 'ModelNet10.zip')

    if osp.exists(osp.join(data_dir, 'meshes')):
        print("ModelNet10 dataset already exists.")
        return
    
    if osp.exists(zip_path):
        print("ModelNet10 zip file already exists.")
    else:
        print("Downloading ModelNet10 dataset...")
        try:
            urllib.request.urlretrieve(url, zip_path, show_progress)
            print("\nDownload complete.")
        except Exception as e:
            raise Exception(f"Failed to download ModelNet10 dataset: {e}")

    print("Extracting ModelNet10 dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)

        mesh_path = osp.join(data_dir, 'meshes')
        if not osp.exists(mesh_path):
            os.makedirs(mesh_path)

        move_off_files(osp.join(data_dir, 'ModelNet10'), mesh_path)
        print("Extraction complete.")
    except Exception as e:
        raise Exception(f"Failed to extract ModelNet10 dataset: {e}")
    finally:
        shutil.rmtree(osp.join(data_dir, 'ModelNet10'))

def to_pcd(off_path, num_points):
    """Convert an OFF file to a point cloud with a given number of points."""
    mesh = o3d.io.read_triangle_mesh(off_path)
    return mesh.sample_points_uniformly(number_of_points=num_points)

def load_model(config):
    """Load the model from the Hugging Face Hub."""
    model = models.make_model(config)
    model_name = "OpenShape/openshape-pointbert-vitg14-rgb"
    
    checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
    pattern = re.compile(r'^module\.')
    key_mapping = {
        'pp_transformer.lift.0.weight': 'pp_transformer.conv.weight',
        'pp_transformer.lift.0.bias': 'pp_transformer.conv.bias',
        'pp_transformer.lift.2.weight': 'pp_transformer.norm.weight',
        'pp_transformer.lift.2.bias': 'pp_transformer.norm.bias',
        'proj.weight': 'head.weight',
        'proj.bias': 'head.bias'
    }
    
    model_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        new_key = re.sub(pattern, '', k).replace('ppat', 'pp_transformer')
        if new_key in key_mapping:
            new_key = key_mapping[new_key]
        model_dict[new_key] = v

    model.load_state_dict(model_dict)
    return model

def preprocess_modelnet10(model, configs, y_up=True):
    """Convert all OFF files in the dataset to point cloud files and extract features."""
    off_files = glob(osp.join(configs.data_dir, 'meshes', '*.off'), recursive=True)
    off_files.sort()
    print(f"Number of .off files: {len(off_files)}")

    embedded_features = {}
    os.makedirs('./modelnet_embed', exist_ok=True)
    for off_file in tqdm(off_files, desc='Converting the dataset type'):
        pcd = to_pcd(off_file, configs.num_points)
        xyz = np.asarray(pcd.points)
        rgb = np.asarray(pcd.colors)
        if y_up:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        xyz = normalize_pc(xyz)

        if rgb is None or len(rgb) == 0:
            rgb = np.ones_like(xyz) * 0.4

        features = np.concatenate([xyz, rgb], axis=1)
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32).unsqueeze(0).cuda()
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).cuda()

        shape_feature = model(xyz_tensor, features_tensor, quantization_size=config.model.voxel_size)
        embedded_features[off_file] = shape_feature.cpu().detach().numpy()[0]

        with open('./modelnet_embed/modelnet.pkl', 'wb') as f:
            pickle.dump(embedded_features, f)

    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and preprocess ModelNet10 dataset')
    parser.add_argument('--download_dir', type=str, default='./data', help='Directory to download ModelNet10 dataset')
    args = parser.parse_args()

    config = load_config()

    model = load_model(config)
    model.cuda().eval()
    download_modelnet10(args.download_dir)
    preprocess_modelnet10(model, config)
    model.cpu()
