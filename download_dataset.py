import os
import urllib.request
import zipfile


def download_modelnet40(args):
    data_dir = './data'
    if args.download_dir:
        data_dir = args.download_dir
        
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    zip_path = os.path.join(data_dir, 'ModelNet40.zip')
    extract_path = os.path.join(data_dir, 'ModelNet40')

    if os.path.exists(extract_path):
        print("ModelNet40 dataset already downloaded and extracted.")
        return data_dir

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
        print("Extraction complete.")
    except Exception as e:
        raise Exception(f"Failed to extract ModelNet40 dataset: {e}")

    os.remove(zip_path)
    print("Zip file removed.")

    return data_dir