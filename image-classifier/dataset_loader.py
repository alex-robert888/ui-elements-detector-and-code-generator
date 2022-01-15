import os
import matplotlib.pyplot as plt
import numpy
import cv2
from configuration import LABELS, LABELS_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, DATASET_ROOT_DIRECTORY
from sklearn.model_selection import train_test_split
from PIL import Image
import timeit

class DatasetLoader(object):
    def __init__(self, should_split_dataset: bool = False) -> None:
        self.__should_split_dataset = should_split_dataset
        self.__train_images = numpy.array([])
        self.__train_labels = numpy.array([])
        self.__test_images = numpy.array([])
        self.__test_labels = numpy.array([])

    def call(self) -> tuple[numpy.array, numpy.array, numpy.array, numpy.array]:
        if self.__should_split_dataset \
                or not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
            self.__split_data()

        self.__train_images, self.__train_labels, self.__test_images, self.__test_labels = self.__load_data()
        self.__plot_some_images(self.__train_images, self.__train_labels)

        return self.__train_images, self.__train_labels, self.__test_images, self.__test_labels

    def __split_data(self):
        unloaded_data = self.__load_paths_and_labels()
        train_unloaded_data, test_unloaded_data = train_test_split(unloaded_data, shuffle=True)

        with open(TRAIN_DATA_PATH, "w") as f_train:
            f_train.writelines(train_unloaded_data)

        with open(TEST_DATA_PATH, "w") as f_test:
            f_test.writelines(test_unloaded_data)

    def __load_paths_and_labels(self):
        with open(LABELS_PATH) as f:
            lines = f.readlines()
        return lines

    def __load_data(self) -> tuple[numpy.array, numpy.array, numpy.array, numpy.array]:
        start = timeit.default_timer()
        train_images, train_labels = self.__load_images_and_labels(TRAIN_DATA_PATH)
        test_images, test_labels = self.__load_images_and_labels(TEST_DATA_PATH)
        print('Time to load images: ', timeit.default_timer() - start)
        return train_images, train_labels, test_images, test_labels

    def __load_images_and_labels(self, file_path) -> tuple[numpy.array, numpy.array]:
        images = []
        labels = []

        with open(file_path, "r") as f:
            while line := f.readline():
                line = line.split(',')
                image_path = f"{DATASET_ROOT_DIRECTORY}/{line[0]}"
                images.append(cv2.imread(image_path))
                labels.append(line[1])

        stop = timeit.default_timer()
        return numpy.array(images), numpy.array(labels)

    def __plot_some_images(self, images, labels):
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i])
            plt.xlabel(labels[i])
        plt.show()
