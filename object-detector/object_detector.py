from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from image_search import sliding_window
from image_search import image_pyramid
from img_classifier_model_loader import ImgClassifierModelLoader
from config import BEST_IMAGE_CLASSIFIER_PATH, TEST_IMAGE_PATH
import numpy as np
import argparse
import imutils
import time
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


class ObjectDetector(object):
    def __init__(self):
        self.__image_classifier_model = self.__load_image_classifier_model()
        self.__test_image, self.__test_image_width , self.__test_image_height = self.__load_test_image()
        self.__plot_test_image()

    def run(self):
        pass

    def __load_image_classifier_model(self):
        img_classifier_model_loader = ImgClassifierModelLoader("../image-classifier/77-model/training_resnet50")
        model = img_classifier_model_loader.call()
        return model

    def __load_test_image(self):
        test_image = cv2.imread(TEST_IMAGE_PATH)
        (h, w) = test_image.shape[:2]
        return test_image, w, h

    def __plot_test_image(self):
        plt.imshow(self.__test_image)
        plt.show()
