import keras
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

# Code taken from https://github.com/BIGBALLON/cifar-10-cnn
class LeNet:
    def __init__(self, load_weights=True):
        self.name               = 'lenet'
        self.model_filename     = 'networks/models/lenet.h5'
        self.num_classes        = 10
        self.input_shape        = 32, 32, 3
        self.batch_size         = 1

        if load_weights:
            try:
                self._model = load_model(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay), input_shape=self.input_shape))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay) ))
        model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay) ))
        model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay) ))
        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    # code used from https://github.com/Hyperparticle/one-pixel-attack-keras
    def color_process(self, imgs):
        if imgs.ndim < 4:
            imgs = np.array([imgs])
        imgs = imgs.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for img in imgs:
            for i in range(3):
                img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
        return imgs

    def predict(self, img):
        processed = self.color_process(img)
        return self._model.predict(processed, batch_size=self.batch_size)

    def predict_one(self, img):
        return self.predict(img)[0]

