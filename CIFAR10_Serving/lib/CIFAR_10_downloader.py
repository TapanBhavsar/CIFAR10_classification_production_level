from dataset_downloader import *

import tarfile
import shutil
import glob


class CIFAR10Downloader(DatasetDownloader):
    def __init__(self, URL, path):
        self.url = URL
        self.path = path

    def extract_downloaded_dataset(self):
        self.download_dataset()
        for file in glob.glob(self.path + "*.tar.gz"):
            shutil.unpack_archive(file, extract_dir=self.path)

