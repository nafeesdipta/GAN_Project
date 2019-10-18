'''
Collected from GAN Cookbook

'''
import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model
import keras.backend as K

#WGAN
def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)

class Discriminator(object):

    def __init__(self, width = 28, height = 28, channels =1, latent_size = 100):
        
        self.CAPACITY = width*height*channels
        self.SHAPE = (width,height,channels)
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)
        self.Discriminator = self.model()

        #self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'] )”
        #WGAN Loss Function
        self.Discriminator.compile(loss='wasserstein_loss', optimizer=self.OPTIMIZER, metrics=['accuracy'] )

        self.Discriminator.summary()

    def model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.SHAPE))
        model.add(Dense(self.CAPACITY, input_shape=self.SHAPE))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(int(self.CAPACITY/2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def summary(self):
        return self.Discriminator.summary()

    def save_model(self):
        plot_model(self.Discriminator.model, to_file='/Users/masnoonnafees/Documents/GAN/Discriminator_Model.png')

    #print (summary)