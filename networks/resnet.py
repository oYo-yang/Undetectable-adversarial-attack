import keras
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras import regularizers

# Code taken from https://github.com/BIGBALLON/cifar-10-cnn
class ResNet:
    def __init__(self, load_weights=True):
        self.name               = 'resnet'
        self.model_filename     = 'networks/models/resnet.h5'
        
        self.stack_n            = 5    
        self.num_classes        = 10
        self.img_rows, self.img_cols = 32, 32
        self.img_channels       = 3
        self.batch_size         = 1

        if load_weights:
            try:
                self._model = load_model(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)

    def residual_network(self, img_input,classes_num=10,stack_n=5):
        def residual_block(intput,out_channel,increase=False):
            if increase:
                stride = (2,2)
            else:
                stride = (1,1)

            pre_bn   = BatchNormalization()(intput)
            pre_relu = Activation('relu')(pre_bn)

            conv_1 = Conv2D(out_channel,kernel_size=(3,3),strides=stride,padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(self.weight_decay))(pre_relu)
            bn_1   = BatchNormalization()(conv_1)
            relu1  = Activation('relu')(bn_1)
            conv_2 = Conv2D(out_channel,kernel_size=(3,3),strides=(1,1),padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(self.weight_decay))(relu1)
            if increase:
                projection = Conv2D(out_channel,
                                    kernel_size=(1,1),
                                    strides=(2,2),
                                    padding='same',
                                    kernel_initializer="he_normal",
                                    kernel_regularizer=regularizers.l2(self.weight_decay))(intput)
                block = add([conv_2, projection])
            else:
                block = add([intput,conv_2])
            return block

        # build model
        # total layers = stack_n * 3 * 2 + 2
        # stack_n = 5 by default, total layers = 32
        # input: 32x32x3 output: 32x32x16
        x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(img_input)

        # input: 32x32x16 output: 32x32x16
        for _ in range(stack_n):
            x = residual_block(x,16,False)

        # input: 32x32x16 output: 16x16x32
        x = residual_block(x,32,True)
        for _ in range(1,stack_n):
            x = residual_block(x,32,False)
        
        # input: 16x16x32 output: 8x8x64
        x = residual_block(x,64,True)
        for _ in range(1,stack_n):
            x = residual_block(x,64,False)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)

        # input: 64 output: 10
        x = Dense(classes_num,activation='softmax',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        return x

    # functions used from https://github.com/Hyperparticle/one-pixel-attack-keras
    def color_process(self, imgs):
        if imgs.ndim < 4:
            imgs = np.array([imgs])
        imgs = imgs.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for img in imgs:
            for i in range(3):
                img[:,:,i] = (img[:,:,i] - mean[i]) / std[i]
        return imgs

    def predict(self, img):
        processed = self.color_process(img)
        return self._model.predict(processed, batch_size=self.batch_size)

    def predict_one(self, img):
        return self.predict(img)[0]
