from CIFAR_10_downloader import CIFAR10Downloader
from CIFAR_10_dataset import CIFAR10Dataset
from data_preprocessor import DataPreprocessor


def main():
    DATASET_PATH = "./CIFAR10_Serving/dataset/"
    DATASET_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    # cifar10 = CIFAR10Downloader(URL=DATASET_URL, path=DATASET_PATH)
    # cifar10.extract_downloaded_dataset()

    dataset = CIFAR10Dataset(path=DATASET_PATH)
    train_data, train_labels, test_data, test_labels = dataset.create_dataset_format()

    preprocessor = DataPreprocessor(data=train_data, labels=train_labels)
    preprocessor.store_preprocessing_parameters()
    train_data = preprocessor.get_reprocessed_data()
    train_labels = preprocessor.one_hot_encode_labels()

    print("first image: ")
    print(train_data[0])
    print("train_data shape: ", train_data.shape)
    print("train_labels shape: ", train_labels.shape)
    # print("test_data shape: ", test_data.shape)
    # print("test_labels shape: ", test_labels.shape)


if __name__ == "__main__":
    main()
