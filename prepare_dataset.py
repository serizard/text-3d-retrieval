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
    with tqdm(total=len(off_files), desc='Moving datasets') as pbar:
        for file_path in off_files:
            dest_file_path = osp.join(dest_dir, osp.basename(file_path))
            shutil.move(file_path, dest_file_path)
            pbar.update(1)

def download_modelnet40(data_dir):
    """Download and extract the ModelNet40 dataset."""
    url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    zip_path = osp.join(data_dir, 'ModelNet40.zip')

    print("Downloading ModelNet40 dataset...")
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        progress = downloaded / total_size * 100
        print(f"\rProgress: {progress:.2f}%", end="")

    try:
        urllib.request.urlretrieve(url, zip_path, show_progress)
        print("\nDownload complete.")
    except Exception as e:
        raise Exception(f"Failed to download ModelNet40 dataset: {e}")

    print("Extracting ModelNet40 dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)

        mesh_path = osp.join(data_dir, 'meshes')
        if not osp.exists(mesh_path):
            os.makedirs(mesh_path)

        move_off_files(osp.join(data_dir, 'ModelNet40'), mesh_path)
        print("Extraction complete.")
    except Exception as e:
        raise Exception(f"Failed to extract ModelNet40 dataset: {e}")
    finally:
        shutil.rmtree(osp.join(data_dir, 'ModelNet40'))

def to_pcd(off_path, num_points):
    """Convert an OFF file to a point cloud with a given number of points."""
    mesh = o3d.io.read_triangle_mesh(off_path)
    return mesh.sample_points_uniformly(number_of_points=num_points)

def load_model(config):
    """Load the model from the Hugging Face Hub."""
    model = models.make_model(config)
    model_name = "OpenShape/openshape-pointbert-vitg14-rgb"

    checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in checkpoint['state_dict'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
    model.load_state_dict(model_dict)
    return model

def preprocess_modelnet40(model, configs, y_up=True):
    """Convert all OFF files in the dataset to point cloud files and extract features."""
    off_files = glob(osp.join(configs.data_dir, 'meshes', '*.off'), recursive=True)
    off_files.sort()
    print(f"Number of .off files: {len(off_files)}")

    embedded_features = {}
    os.makedirs('./modelnet_embed', exist_ok=True)
    for off_file in tqdm(off_files, desc='Converting the dataset type'):
        if configs.preprocessing_ckpt is not None and off_file < configs.preprocessing_ckpt:
            continue
        pcd = to_pcd(off_file, configs.num_points)
        xyz = np.asarray(pcd.points)
        rgb = np.asarray(pcd.colors)
        if y_up:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        xyz = normalize_pc(xyz)

        if rgb is None:
            rgb = np.ones_like(xyz) * 0.4

        features = np.concatenate([xyz, rgb], axis=1)
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32)
        features_tensor = torch.tensor(features, dtype=torch.float32)

        shape_feature = model(xyz_tensor, features_tensor, device='cuda', quantization_size=config.model.voxel_size)
        embedded_features[off_file] = shape_feature.cpu().numpy()[0]

        with open('./modelnet_embed/modelnet.pkl', 'wb') as f:
            pickle.dump(embedded_features, f)

    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and preprocess ModelNet40 dataset')
    parser.add_argument('--download_dir', type=str, default='./data', help='Directory to download ModelNet40 dataset')
    args = parser.parse_args()

    config = load_config()

    model = load_model(config)
    download_modelnet40(args.download_dir)
    preprocess_modelnet40(model, config)
