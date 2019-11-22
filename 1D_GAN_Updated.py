import numpy as np 
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from numpy import hstack
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense, Flatten
from matplotlib import pyplot
import keras.backend as K
from keras.utils.vis_utils import plot_model
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.spatial.distance import pdist
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#WGAN
def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)

# define the standalone discriminator model
def define_discriminator(n_inputs=10):
    model = Sequential()
    #model.add(Flatten())
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_shape=(n_inputs, 1)))
    #model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim, n_outputs=1):
    model = Sequential()
    #model.add(Flatten())
    #print(latent_dim)
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_shape=(latent_dim, 1)))
    #model.add(Flatten())
    model.add(Dense(n_outputs, activation='linear'))
    return model

##Second generator

def define_generator2(latent_dim, n_outputs=1):
    model = Sequential()
    #model.add(Flatten())
    #print(latent_dim)
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_shape=(latent_dim, 1)))
    #model.add(Flatten())
    model.add(Dense(n_outputs, activation='linear'))
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, generator2, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the second generator
    model.add(generator2)
    model.add(discriminator)
    # compile model
    #model.add(Flatten())
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    #model.summary()
    return model



def generate_real(generator, x):
    # generate random Gaussian values
    # seed random number generator
    seed(5)
    x = int (x)
    # generate some Gaussian values
    values = randn(x) - 3
    #y = ones((x, 1))
    
    #print (values.shape)
    #y = y.reshape(x, 1)
    #b = np.hstack((values, np.ones((values.shape[0], 1), dtype= object)))
    #ap = hstack(values, y)
    #b = b.reshape(x, 2)
    #b[:,b.shape[1]-1] = int(1)
    #np.savetxt('real_1d.csv', b, delimiter=',', fmt="%s")
    values = values.reshape(-1,1)
    nn = NearestNeighbors(size, metric='cosine', algorithm='brute').fit(values)
    dists, idxs = nn.kneighbors(values)
    print("Nearest Point for K=1 real", idxs)
    pdis = pdist(values, dis)
    values = values.reshape(1, x, 1)
    values = generator.predict(values)
    plt.scatter(values, values, marker = '^')
    plt.scatter(pdis, pdis, marker='*')
    plt.show()
    y = ones((x, 1))
    return values, y

def generate_fake(generator2, x):
    # generate random Gaussian values
    # seed random number generator
    seed(10)
    x = int (x)
    # generate some Gaussian values
    values = randn(x) + 1
    #y = []
    
    
    #print (values.shape)
    #y = y.reshape(x, 1)
    #b = np.hstack((values, np.zeros((values.shape[0], 1), dtype =object)))
    #b = b.reshape(x, 2)
    #b[:,b.shape[1]-1] = int(0)
    #ap = hstack(values, y)
    #np.savetxt('fake_1d.csv', b, delimiter=',', fmt="%s")
    #print (ap)
    values = values.reshape(-1,1)
    nn = NearestNeighbors(size, metric='cosine', algorithm='brute').fit(values)
    dists, idxs = nn.kneighbors(values)
    print("Nearest Point for K=1 fake", idxs)
    pdis = pdist(values, dis)
    values = values.reshape(1, x, 1)
    values = generator2.predict(values)
    plt.scatter(values, values, marker = 'o')
    plt.scatter(pdis, pdis, marker='|')
    #plt.scatter(generate_real(size), generate_real(size), marker= 'o')
    plt.show()
    #print(values)
    y = zeros((x, 1))
    
    return values, y

'''
def plot_values(size):
    
    x = generate_fake(size)
    y = generate_real(size)

    plt.scatter(generate_fake(size), generate_fake(size), marker = '^')
    plt.scatter(generate_real(size), generate_real(size), marker= 'o')
    plt.show()
    plot_values(size)

'''
# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, generator2, discriminator, n):
    # prepare real samples
    x_real, y_real = generate_real(generator, n)
    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake(generator2, n)
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print(epoch, acc_real, acc_fake)
    # scatter plot real and fake data points
    pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
    pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    pyplot.show()

# train the generator and discriminator
def train(g_model, g_model2, d_model, gan_model, size, n_epochs=3, n_eval=2):
    # determine half the size of one batch, for updating the discriminator
    half_batch = size
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        x_real, y_real = generate_real(g_model, half_batch)
        #x_real = x_real
        # prepare fake examples
        x_fake, y_fake = generate_fake(g_model2, half_batch)
        # update discriminator
        
        x_real= x_real.reshape(1, size,1)
        #print("xreal", x_real)
        y_real = y_real.reshape(1, size, 1)
        x_fake = x_fake.reshape(1, size, 1)
        y_fake = y_fake.reshape(1, size, 1)
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # prepare points in latent space as input for the generator
        #x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        #y_gan = ones((n_batch, 1))
       # xx_gan = generate_latent_points(latent_dim, n_batch)
       # yy_gan = ones((n_batch, 0))
        # update the generator via the discriminator's error
        x_real = x_real.reshape(-1,1)
        nn = NearestNeighbors(size, metric='cosine', algorithm='brute').fit(x_real)
        dists, idxs = nn.kneighbors(x_real)
        pdis_r = pdist(x_real, dis)
        #plt.scatter(dists, dists, marker='*')
        x_fake = x_fake.reshape(-1,1)
        nn = NearestNeighbors(size, metric='cosine', algorithm='brute').fit(x_fake)
        dists, idxs = nn.kneighbors(x_fake)
        #plt.scatter(dists, dists, marker='|')
        pdis_f = pdist(x_fake, dis)
        plt.scatter(x_real, x_real, marker = '^')
        plt.scatter(x_fake, x_fake, marker = 'o')
        plt.scatter(pdis_f, pdis_f, marker='|')
        plt.scatter(pdis_r, pdis_r, marker='*')
        plt.show()
        #gan_model.train_on_batch(x_real, y_real)
        #gan_model.train_on_batch(x_fake, y_fake)
        # evaluate the model every n_eval epochs
        
        #if (i+1) % n_eval == 0:
           #summarize_performance(i, g_model, g_model2, d_model, size)

# size of the latent space
size = 10
dis = 'cosine'
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(size)

generator2 = define_generator2(size)
# create the gan

gan_model = define_gan(generator, generator2, discriminator)
# train model

gan_model.summary()
# plot gan model
#plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)

train(generator, generator2, discriminator, gan_model, size)