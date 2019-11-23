from CIFAR_10_downloader import CIFAR10Downloader
from CIFAR_10_dataset import CIFAR10Dataset


def main():
    path = "./CIFAR10_Serving/dataset/"
    cifar10 = CIFAR10Downloader(
        URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", path=path,
    )
    cifar10.extract_downloaded_dataset()

    dataset = CIFAR10Dataset(path=path)
    dataset.create_dataset_format()


if __name__ == "__main__":
    main()
