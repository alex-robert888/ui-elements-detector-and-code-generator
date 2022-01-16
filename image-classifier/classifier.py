import timeit
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from configuration import LABELS_LIST, IMAGE_SIZE, CALLBACK_CHECKPOINT_PATH, MODEL_SAVE_PATH
import matplotlib.pyplot as plt
from datetime import datetime


class Classifier(object):
    def __init__(self, train_images, train_labels, test_images, test_labels) -> None:
        self.__num_classes = len(LABELS_LIST)
        self.__train_images = train_images
        self.__train_labels = train_labels
        self.__test_images = test_images
        self.__test_labels = test_labels
        self.__model = None
        self.__history = None
        self.__epochs_count = 10

    def train_and_test(self):
        print("Start training.")
        start = timeit.default_timer()
        self.__model = self.__create_model()
        self.__compile_model()
        self.__model.summary()
        self.__history = self.__train_model()
        print("Finished training: ", timeit.default_timer() - start)

        self.__model.save(MODEL_SAVE_PATH)

        print("Evaluate the model")
        self.__evaluate_model()

    ########## DIFFERENT MODELS ###########
    def __create_model(self):
        return Sequential([
            layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.__num_classes)
        ])

    #### Transfer learning using ResNet50
    def __create_model_resnet50(self):
        inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        resnet50_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
        )

        # freeze all layers except for the last block
        for layer in resnet50_model.layers[:143]:
            layer.trainable = False

        for i, layer in enumerate(resnet50_model.layers):
            print(i, layer.name, layer.trainable)

        # create new sequential model to which we add the resnet layers
        model = keras.models.Sequential()
        model.add(resnet50_model)
        model.add(keras.layers.Flatten())

        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.__num_classes, activation='softmax'))

        return model
    ######################################

    def __compile_model(self):
        self.__model.compile(optimizer='adam',
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])

    def __train_model(self):
        checkpoint_path = CALLBACK_CHECKPOINT_PATH

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        return self.__model.fit(
            self.__train_images, self.__train_labels,
            epochs=self.__epochs_count,
            validation_data=(self.__test_images, self.__test_labels),
            callbacks=[cp_callback]
        )

    def __evaluate_model(self):
        acc = self.__history.history['accuracy']
        val_acc = self.__history.history['val_accuracy']
        loss = self.__history.history['loss']
        val_loss = self.__history.history['val_loss']

        epochs_range = range(self.__epochs_count)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()