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
import tensorflow as tf
from statistics import mean 

os.environ['KMP_DUPLICATE_LIB_OK']='True'


#WGAN
def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)

def custom_loss(g1, g2):
    return (g1-g2)

def custom_loss_gan1(g1, g2):
    dis_loss = g1-g2
    #if dis_loss <0:
        #dis_loss = dis_loss * -1
    #return (g1_loss-g2_loss)
    print("g111: ")
    return (tf.maximum(g1, g2))

def custom_loss_gan2(g1, g2):
    dis_loss = g1-g2
    #if dis_loss>0:
        #dis_loss = dis_loss * -1
    #return (g1_loss-g2_loss)
    return (tf.minimum(g1,g2))

# define the standalone discriminator model
def define_discriminator(n_inputs=100):
    model = Sequential()
    #model.add(Flatten())
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_shape=(n_inputs, 1)))
    #model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim, n_outputs=1):
    model = Sequential()
    #model.add(Flatten())
    #print(latent_dim)
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_shape=(latent_dim, 1)))
    #model.add(Flatten())
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss=custom_loss_gan1, optimizer='adam', metrics=['accuracy'])
    return model

##Second generator

def define_generator2(latent_dim, n_outputs=1):
    model = Sequential()
    #model.add(Flatten())
    #print(latent_dim)
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_shape=(latent_dim, 1)))
    #model.add(Flatten())
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss=custom_loss_gan2, optimizer='adam', metrics=['accuracy'])
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, generator2, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    #model.compile(loss=custom_loss_gan1, optimizer='adam', metrics=['accuracy'])
    # add the second generator
    model.add(generator2)
    #model.compile(loss=custom_loss_gan2, optimizer='adam', metrics=['accuracy'])
    model.add(discriminator)
    # compile model
    #model.add(Flatten())
    model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])
    
    #model.summary()
    return model



def generate_real(generator, x):
    # generate random Gaussian values
    # seed random number generator
    seed(10)
    x = int (x)
    # generate some Gaussian values
    values = randn(x) - 1
    #y = ones((x, 1))
    
    #print (values.shape)
    #y = y.reshape(x, 1)
    #b = np.hstack((values, np.ones((values.shape[0], 1), dtype= object)))
    #ap = hstack(values, y)
    #b = b.reshape(x, 2)
    #b[:,b.shape[1]-1] = int(1)
    #np.savetxt('real_1d.csv', b, delimiter=',', fmt="%s")
    
    #print("Nearest Point for real", pdis)
    values = values.reshape(1, x, 1)
    y = ones((x, 1))
    y = y.reshape(1, x, 1)
    generator.fit(values, y, epochs=100)
    
    values = generator.predict(values)
    values = values.reshape(-1,1)
    nn = NearestNeighbors(batch_size, metric=dis, algorithm='brute').fit(values)
    dists, idxs = nn.kneighbors(values)
    pdis = pdist(values, dis)
    #r_nn.append(pdis)
    plt.scatter(values, values, marker = '^')
    plt.scatter(pdis, pdis, marker='*')
    #plt.show()
    plt.close()
    
    return values, y

def generate_fake(generator2, x):
    # generate random Gaussian values
    # seed random number generator
    seed(5)
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
    
    
    #print("Nearest Point for fake", pdis)
    values = values.reshape(1, x, 1)
    y = zeros((x, 1))
    y = y.reshape(1, x, 1)
    generator2.fit(values, y, epochs=100)
    values = generator2.predict(values)
    values = values.reshape(-1,1)
    nn = NearestNeighbors(batch_size, metric=dis, algorithm='brute').fit(values)
    dists, idxs = nn.kneighbors(values)
    pdis = pdist(values, dis)
    #f_nn.append(pdis)
    plt.scatter(values, values, marker = 'o')
    plt.scatter(pdis, pdis, marker='|')
    #plt.scatter(generate_real(size), generate_real(size), marker= 'o')
    #plt.show()
    #plt.savefig('plot_line_plot_loss.png')
    plt.close()
    #print(values)
    
    
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
def train(g_model, g_model2, d_model, gan_model, batch_size, size, n_epochs=15, n_eval=2):
    # determine half the size of one batch, for updating the discriminator
    #batch_size = 100
    batch_per_epoch = int(size/batch_size)
    
    # manually enumerate epochs
    #g1_tmp, g2_tmp = list(), list()
    for i in range(n_epochs):
        # prepare real samples
        for j in range(batch_per_epoch):
            x_real, y_real = generate_real(g_model, batch_size)

            x_fake, y_fake = generate_fake(g_model2, batch_size)
         # update discriminator
        
            x_real= x_real.reshape(1, batch_size,1)
            y_real = y_real.reshape(1, batch_size, 1)
            x_fake = x_fake.reshape(1, batch_size, 1)
            y_fake = y_fake.reshape(1, batch_size, 1)
        #d_model.train_on_batch(x_real, y_real)
        #d_model.train_on_batch(x_fake, y_fake)
        # prepare points in latent space as input for the generator
        #x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        #y_gan = ones((n_batch, 1))
       # xx_gan = generate_latent_points(latent_dim, n_batch)
       # yy_gan = ones((n_batch, 0))
        # update the generator via the discriminator's error
            x_real = x_real.reshape(-1,1)
            nn = NearestNeighbors(batch_size, metric=dis, algorithm='brute').fit(x_real)
            dists, idxs = nn.kneighbors(x_real)
            pdis_r = pdist(x_real, dis)
            #plt.scatter(dists, dists, marker='*')
            x_fake = x_fake.reshape(-1,1)
            nn = NearestNeighbors(batch_size, metric=dis, algorithm='brute').fit(x_fake)
            dists, idxs = nn.kneighbors(x_fake)
            #plt.scatter(dists, dists, marker='|')
            pdis_f = pdist(x_fake, dis)
            plt.scatter(x_real, x_real, marker = '^')
            plt.scatter(x_fake, x_fake, marker = 'o')
            plt.scatter(pdis_f, pdis_f, marker='|')
            plt.scatter(pdis_r, pdis_r, marker='*')
            #plt.show()
            plt.close()
            f_nn.append(mean(pdis_f))
            r_nn.append(mean(pdis_r))
            x_real= x_real.reshape(1, batch_size,1)
            y_real = y_real.reshape(1, batch_size, 1)
            x_fake = x_fake.reshape(1, batch_size, 1)
            y_fake = y_fake.reshape(1, batch_size, 1)
        
            y1_pred = g_model.predict(x_real)
            y2_pred = g_model2.predict(x_fake)
            g1_loss = g_model.train_on_batch(x_real, y1_pred)
            g2_loss = g_model2.train_on_batch(x_fake, y2_pred)
            #dis_loss = g1_loss[0] - g2_loss[0]

            #g1_loss = g_model.train_on_batch(x_real, y_real)
            #g2_loss = g_model2.train_on_batch(x_fake, y_fake)
            dis_loss = d_model.train_on_batch(x_real, y_real)
            dis_loss = d_model.train_on_batch(x_fake, y_fake)
            #dis_loss = g1_loss[0] - g2_loss[0]
            c1_hist.append(dis_loss[0])
            g1_hist.append(g1_loss)
            g2_hist.append(g2_loss)
            final_plot(r_nn, f_nn, c1_hist)
        #print("g1: ", g1_loss, "g2: ", g2_loss, "d:", dis_loss)
        # evaluate the model every n_eval epochs
        
        #if (i+1) % n_eval == 0:
           #summarize_performance(i, g_model, g_model2, d_model, size)
    plot_history(c1_hist,g1_hist,g2_hist)
    #final_plot(r_nn, f_nn, c1_hist)
# size of the latent space
# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, g1_hist, g2_hist):
	# plot history
	pyplot.plot(d1_hist[0], label='discriminator')
	pyplot.plot(g1_hist[0], label='generator_real')
	pyplot.plot(g2_hist[0], label='generator_fake')
	pyplot.legend()
	#pyplot.savefig('plot_line_plot_loss.png')
	pyplot.show()

def final_plot(r_nn, f_nn, c1_hist):
    pyplot.scatter(r_nn, c1_hist, color='green')

    pyplot.scatter(f_nn, c1_hist, color='blue')
    #pyplot.plot('Nearest Neigbour','discriminator score')
    pyplot.show()
    pyplot.close()


size = 1000
batch_size = int(size/10)
dis = 'euclidean'
g1_loss = 0
g2_loss = 0
dis_loss = 0
c1_hist, g1_hist, g2_hist = [], [], []
r_nn, f_nn = [], []


# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(batch_size)

generator2 = define_generator2(batch_size)
# create the gan

gan_model = define_gan(generator, generator2, discriminator)
# train model

gan_model.summary()
# plot gan model
#plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)

train(generator, generator2, discriminator, gan_model, batch_size, size)
print(c1_hist.count)
print(len(c1_hist))

