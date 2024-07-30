import os
import urllib.request
import zipfile
import shutil


def move_off_files(src_dir, dest_dir):
    # 모든 하위 디렉토리를 포함하여 .off 파일 찾기
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.off'):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_dir, file)
                
                # 파일을 대상 디렉토리로 이동
                shutil.move(src_file_path, dest_file_path)
                print(f"Moved: {src_file_path} -> {dest_file_path}")



def download_modelnet40(args):

    data_dir = args.download_dir
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    zip_path = os.path.join(data_dir, 'ModelNet40.zip')
    extract_path = os.path.join(data_dir, 'ModelNet40')

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

        source_directory = os.path.join(data_dir, 'ModelNet40')
        move_off_files(source_directory, data_dir)
        print("Extraction complete.")
    except Exception as e:
        raise Exception(f"Failed to extract ModelNet40 dataset: {e}")

    os.remove(zip_path)
    print("Zip file removed.")

    return data_dir