import os
import time
import importlib.metadata
from pathlib import Path

from huggingface_hub import snapshot_download
import requests

import gdown
from moviepy.editor import *


def get_project_root() -> Path:
    """
    Get the root directory of the current Python package.
    This function first tries to use the package metadata, and if that fails,
    it falls back to searching for a marker file.
    """
    try:
        if __package__:
            metadata = importlib.metadata.metadata(__package__)
            project_name = metadata["Name"]
            package_path = Path(__package__.replace(".", os.sep))
            return Path(__file__).resolve().parents[len(Path(__file__).parts) - package_path.parts.index(project_name) - 1]
        else:
            # If __package__ is None (script is run directly), fall back to find_project_root
            return find_project_root()
    except (importlib.metadata.PackageNotFoundError, ValueError, IndexError):
        # If any error occurs, fall back to the directory-based approach
        return find_project_root()

def find_project_root(current_path=__file__, marker_files=('pyproject.toml', 'setup.py', 'requirements.txt')):
    """
    Find the project root directory by searching for marker files.
    
    :param current_path: The path to start the search from (default is the current file)
    :param marker_files: A tuple of files that indicate the root of the project
    :return: The absolute path to the project root
    """
    current_dir = Path(current_path).resolve().parent
    
    while True:
        for marker in marker_files:
            if (current_dir / marker).exists():
                return current_dir
        parent_dir = current_dir.parent
        if parent_dir == current_dir:  # We've reached the root of the file system
            raise FileNotFoundError(f"Could not find project root containing any of {marker_files}")
        current_dir = parent_dir

def print_directory_contents(path):
    """
    Print the contents of the given directory, showing only subdirectories.

    :param path: The path to the directory to print
    """
    for child in Path(path).iterdir():
        if child.is_dir():
            print(child)
    print("\n")

def download_model(checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        print("Checkpoint Not Downloaded, start downloading...")
        tic = time.time()
        snapshot_download(
            repo_id="TMElyralab/MuseTalk",
            local_dir=checkpoints_dir,
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        # weight
        os.makedirs(f"{checkpoints_dir}/sd-vae-ft-mse/")
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=checkpoints_dir+'/sd-vae-ft-mse',
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        #vae
        url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
        response = requests.get(url)
        # 确保请求成功
        if response.status_code == 200:
            # 指定文件保存的位置
            file_path = f"{checkpoints_dir}/whisper/tiny.pt"
            os.makedirs(f"{checkpoints_dir}/whisper/")
            # 将文件内容写入指定位置
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"请求失败，状态码：{response.status_code}")
        #gdown face parse
        url = "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
        os.makedirs(f"{checkpoints_dir}/face-parse-bisent/")
        file_path = f"{checkpoints_dir}/face-parse-bisent/79999_iter.pth"
        gdown.download(url, file_path, quiet=False)
        #resnet
        url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        response = requests.get(url)
        # 确保请求成功
        if response.status_code == 200:
            # 指定文件保存的位置
            file_path = f"{checkpoints_dir}/face-parse-bisent/resnet18-5c106cde.pth"
            # 将文件内容写入指定位置
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"请求失败，状态码：{response.status_code}")


        toc = time.time()

        print(f"download cost {toc-tic} seconds")
        print_directory_contents(checkpoints_dir)

    else:
        print("Already download the model.")



# Usage example
if __name__ == "__main__":
    project_dir = get_project_root()
    print(f"Project root: {project_dir}")

    checkpoints_dir = project_dir / "models"
    print(f"Checkpoints directory: {checkpoints_dir}")

    print("\nDirectory contents:")
    print_directory_contents(project_dir)
    
    download_model(checkpoints_dir)  # for huggingface deployment.

