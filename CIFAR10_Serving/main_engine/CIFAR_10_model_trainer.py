from CIFAR_10_downloader import CIFAR10Downloader
from CIFAR_10_dataset import CIFAR10Dataset
from data_preprocessor import DataPreprocessor
from sklearn.model_selection import train_test_split
from cifar10_model_trainer import CIFAR10ModelTrainer


def main():
    DATASET_PATH = "./CIFAR10_Serving/dataset/"
    DATASET_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    MODEL_PATH = "./CIFAR10_Serving/model_weights/model.ckpt"
    PARAMETER_FILE_PATH = "./CIFAR10_Serving/parameters/parameters.npy"
    cifar10 = CIFAR10Downloader(URL=DATASET_URL, path=DATASET_PATH)
    cifar10.extract_downloaded_dataset()

    dataset = CIFAR10Dataset(path=DATASET_PATH)
    train_data, train_labels, test_data, test_labels = dataset.create_dataset_format()

    preprocessor = DataPreprocessor(data=train_data, labels=train_labels)
    preprocessor.store_preprocessing_parameters(file_name=PARAMETER_FILE_PATH)
    train_data = preprocessor.get_reprocessed_data()
    train_labels = preprocessor.one_hot_encode_labels()

    preprocessor_test = DataPreprocessor(data=test_data, labels=test_labels)
    preprocessor_test.restore_preprocessing_parameters(file_name=PARAMETER_FILE_PATH)
    test_data = preprocessor_test.get_reprocessed_data()
    test_labels = preprocessor_test.one_hot_encode_labels()

    train_data, validation_data, train_labels, validation_labels = train_test_split(
        train_data, train_labels, test_size=0.2
    )
    print("train_data shape: ", train_data.shape)
    print("train_labels shape: ", train_labels.shape)

    print("validation_data shape: ", validation_data.shape)
    print("validation_labels shape: ", validation_labels.shape)

    print("test_data shape: ", test_data.shape)
    print("test_labels shape: ", test_labels.shape)

    model_trainer = CIFAR10ModelTrainer(
        train_data=train_data,
        train_labels=train_labels,
        validation_data=validation_data,
        validation_labels=validation_labels,
        test_data=test_data,
        test_labels=test_labels,
    )
    model_trainer.train_model(epochs=10, batch_size=32)
    model_trainer.save_model(model_path=MODEL_PATH)


if __name__ == "__main__":
    main()
