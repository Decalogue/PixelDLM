import os


def ensure_dir(dir_path):
    """确保目录存在"""
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
