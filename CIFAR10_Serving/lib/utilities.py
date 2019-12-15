import os


def create_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
