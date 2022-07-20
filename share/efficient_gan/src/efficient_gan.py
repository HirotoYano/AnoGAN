from tensorflow.keras.layers import Input, Dense, LeakyReLU, Concatenate, Dropout
from tensorflow.keras.models import Model
# from sklearn.preprocessing import minmax_scale

import tensorflow as tf
# import numpy as np


class EfficientGAN():
    def __init__(self, input_dim=0, latent_dim=32):
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

    def get_encoder(self, initializer=tf.keras.initializers.GlorotUniform()):
        inputs = Input(shape=(self.input_dim,), name='input')
        net = Dense(64, activation=LeakyReLU(alpha=0.1), kernel_initializer=initializer, name='layer_1')(inputs)
        outputs = Dense(self.latent_dim, activation='linear', kernel_initializer=initializer, name='output')(net)

        return Model(inputs=inputs, outputs=outputs, name='Encoder')

    def get_generator(self, initializer=tf.keras.initializers.GlorotUniform()):
        inputs = Input(shape=(self.latent_dim,), name='input')
        net = Dense(64, activation='relu', kernel_initializer=initializer, name='layer_2')(inputs)
        outputs = Dense(self.input_dim, activation='linear', kernel_initializer=initializer, name='output')(net)

        return Model(inputs=inputs, outputs=outputs, name='Generator')

    def get_discriminator(self, initializer=tf.keras.initializers.GlorotUniform()):
        # D(x)
        inputs1 = Input(shape=(self.input_dim,), name='real')
        net = Dense(128, activation=LeakyReLU(alpha=0.1), kernel_initializer=initializer, name='layer_1')(inputs1)
        dx = Dropout(.2)(net)

        # D(z)
        inputs2 = Input(shape=(self.latent_dim,), name='noise')
        net = Dense(128, activation=LeakyReLU(alpha=0.1), kernel_initializer=initializer, name='layer_2')(inputs2)
        dz = Dropout(.2)(net)

        # concat D(x) and D(z)
        conet = Concatenate(axis=1)([dx, dz])

        # D(x, z)
        conet = Dense(128, activation=LeakyReLU(alpha=0.1), kernel_initializer=initializer, name='layer_3')(conet)
        conet = Dropout(.2)(conet)
        outputs = Dense(1, activation='sigmoid', kernel_initializer=initializer, name='output')(net)

        return Model(inputs=[inputs1, inputs2], outputs=outputs, name='Discriminator')

