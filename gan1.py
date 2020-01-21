'''
Masnoon Nafees
The implementation idea's has been taken from 
machinelearningmastery.com from Jason
Actual tutorial has more complex method like run MTCNN extract face
and then convert into a npz compressed file Which is time consuming
and complex.
If ValueError: low >= high error, update numpy or use pycharm
'''

'''
For WGAN ADD this into the script
kernel_constraint=const
Source: machinelearningmastery.com
The differences in implementation for the WGAN are as follows:

Use a linear activation function in the output layer of the critic model (instead of sigmoid).
Use -1 labels for real images and 1 labels for fake images (instead of 1 and 0).
Use Wasserstein loss to train the critic and generator models.
Constrain critic model weights to a limited range after each mini batch update (e.g. [-0.01,0.01]).
Update the critic model more times than the generator each iteration (e.g. 5).
Use the RMSProp version of gradient descent with a small learning rate and no momentum (e.g. 0.00005).

For WGAN ADD this into the script
kernel_constraint=const
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value

	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)

	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

'''
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import mean
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
import cv2
import glob
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def define_discriminator():
	model = Sequential()
	#Downsampling per layer by half
	model.add(Conv2D(128, (5,5), padding='same', input_shape=shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def define_generator(latent_dim):
	model = Sequential()
	#Upsampling per layer by half
	n_nodes = 128 * 5 * 5
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((5, 5, 128)))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
	return model

def define_gan(g_model, d_model):
	d_model.trainable = False #Prevent Discrimanotr's loss update before Generator
	model = Sequential()
	model.add(g_model)
	model.add(d_model)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def generate_real_samples(dataset, n_samples):
	ix = randint(1, dataset.shape[0], n_samples) #mean 0 and SD 1
	X = dataset[ix]
	y = ones((n_samples, 1))
	return X, y

def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim * n_samples)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
	x_input = generate_latent_points(latent_dim, n_samples)
	X = g_model.predict(x_input)
	y = zeros((n_samples, 1))
	return X, y

def save_plot(examples, epoch, n=5):
	# scale from [-1,1] to [0,1]
	examples = (examples + 1) / 2.0
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i])
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	X_real, y_real = generate_real_samples(dataset, n_samples)
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	save_plot(x_fake, epoch)

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1000, n_batch=30):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)#15
	for i in range(n_epochs):
		c1_tmp, c2_tmp = list(), list()
		for j in range(bat_per_epo):
			X_real, y_real = generate_real_samples(dataset, half_batch)
			d_loss1, _ = d_model.train_on_batch(X_real, y_real)
			c1_tmp.append(d_loss1)
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
			c2_tmp.append(d_loss2)
			c1_hist.append(mean(c1_tmp))
			c2_hist.append(mean(c2_tmp))
			X_gan = generate_latent_points(latent_dim, n_batch)
			y_gan = ones((n_batch, 1))
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			g_hist.append(g_loss)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
		# evaluate the model performance, sometimes
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)
	plot_history(c1_hist, c2_hist, g_hist)

#Please change the path according to your machine
def load_real_samples1():
	cv_img = []
	for img in glob.glob("/Users/masnoonnafees/Documents/GAN/project/Aaron/*.jpg"):
		n=cv2.imread(img)
		n=cv2.resize(n,(80,80)) #convert into 80*80
		n=n.astype(float)#convert to float
		cv_img.append(n)
	X = np.asarray(cv_img)
	X=X.astype('float32')
	X=(X-127.5)/127.5 #convert into 0 and 1 floating value
	print(X.shape)
	return X

def plot_history(d1_hist, d2_hist, g_hist):
	# plot history
	pyplot.plot(d1_hist, label='dis_real')
	pyplot.plot(d2_hist, label='dis_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig('plot_line_plot_loss.png')
	pyplot.close()

latent_dim = 100
c1_hist, c2_hist, g_hist = list(), list(), list()
shape=(80,80,3)
d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)
dataset = load_real_samples1()
train(g_model, d_model, gan_model, dataset, latent_dim)