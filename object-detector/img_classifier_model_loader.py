from tensorflow import keras
import tensorflow as tf


class ImgClassifierModelLoader(object):
    def __init__(self, checkpoint_path):
        self.__checkpoint_path = checkpoint_path
        self.__model = None
        self.__num_classes = 21

    def call(self):
        self.__model = self.__create_model_resnet50()
        self.__compile_model()
        self.__model.load_weights(self.__checkpoint_path)
        return self.__model

    def __create_model_resnet50(self):
        inputs = keras.Input(shape=(224, 224, 3))
        resnet50_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            input_shape=(224, 224, 3)
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

    def __compile_model(self):
        self.__model.compile(optimizer='adam',
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])

