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
import open_clip
from utils.data import process_input




def correct_off_header(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if len(lines[0].split('OFF')) > 2:
        header = lines[0].split('OFF')
        lines[0] = 'OFF\n'
        lines.insert(1, ' '.join(header[1:]).strip() + '\n')

    with open(file_path, 'w') as file:
        file.writelines(lines)


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
    """Download and extract the ModelNet40 dataset."""
    url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    zip_path = osp.join(data_dir, 'ModelNet40.zip')

    if osp.exists(osp.join(data_dir, 'meshes')):
        print("ModelNet40 dataset already exists.")
        return
    
    if osp.exists(zip_path):
        print("ModelNet40 zip file already exists.")
    else:
        print("Downloading ModelNet40 dataset...")
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
    correct_off_header(off_path)
    try:
        mesh = o3d.io.read_triangle_mesh(off_path)
    except:
        print(f"Failed to read {off_path}")
        return None
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
    for off_file in tqdm(off_files, desc='Encoding the dataset'):
        if osp.exists('./modelnet_embed/modelnet.pkl'):
            with open('./modelnet_embed/modelnet.pkl', 'rb') as f:
                try:
                    embedded_features = pickle.load(f)
                    if off_file in embedded_features.keys():
                        continue
                except:
                    pass
        try:
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
        except:
            print(f"Failed to convert {off_file}")
            continue

    print("Preprocessing complete.")


def embed_keywords(data_dir):
    data_names = np.array(list(glob(osp.join(data_dir, '/meshes/*.off'))))

    open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', 
                                                                                    pretrained='laion2b_s39b_b160k', 
                                                                                    cache_dir='./clip_cache')
    if torch.cuda.is_available():
        open_clip_model = open_clip_model.cuda().eval()
    else:
        open_clip_model = open_clip_model.cpu().eval()

    text_features = {}
    data_names = np.unique([name.split('\\')[1].split('_')[0] for name in data_names])
    
    for data_name in tqdm(data_names):
        text_feature = process_input(data_name, open_clip_model, 'cpu')
        text_features[data_name] = text_feature

    with open('./modelnet_embed/modelnet_name.pkl', 'wb') as f:
        pickle.dump(text_features, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and preprocess ModelNet10 dataset')
    parser.add_argument('--download_dir', type=str, default='./data', help='Directory to download ModelNet40 dataset')
    args = parser.parse_args()

    config = load_config()

    model = load_model(config)
    model.cuda().eval()
    download_modelnet10(args.download_dir)
    preprocess_modelnet10(model, config)
    embed_keywords(args.download_dir)
    model.cpu()
