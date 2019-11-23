import os
import wget


class DatasetDownloader(object):
    def __init__(self, URL, path):
        self.url = URL
        self.path = path

    def create_dataset_directory(self):
        if os.path.isdir(self.path):
            return 0
        else:
            os.mkdir(self.path)
            return 1

    def check_dataset(self):
        if len(os.listdir(self.path)) == 0:
            return 0
        else:
            return 1

    def download_dataset(self):
        self.create_dataset_directory()

        if self.check_dataset():
            print("data is present")
        else:
            filename = wget.download(self.url, out=self.path)
            print(f"{filename} is downloaded")
