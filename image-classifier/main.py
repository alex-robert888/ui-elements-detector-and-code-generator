from dataset_loader import DatasetLoader
from classifier import Classifier


def load_dataset():
    dataset_loader = DatasetLoader()
    train_images, train_labels, test_images, test_labels = dataset_loader.call()
    return train_images, train_labels, test_images, test_labels


def main():
    train_images, train_labels, test_images, test_labels = load_dataset()
    classifier = Classifier(train_images, train_labels, test_images, test_labels)
    classifier.train_and_test()


if __name__ == "__main__":
    main()
