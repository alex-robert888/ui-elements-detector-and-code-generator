class Classifier(object):
    def __init__(self, train_images, train_labels, test_images, test_labels) -> None:
        self.__train_images = train_images
        self.__train_labels = train_labels
        self.__test_images = test_images
        self.__test_labels = test_labels

    def train_and_test(self):
        pass
