import numpy as np 
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from numpy import hstack
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import randn
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Layer,Dense, Flatten, BatchNormalization, LeakyReLU
from matplotlib import pyplot
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import  RMSprop
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.spatial.distance import pdist
import os

from statistics import mean 


os.environ['KMP_DUPLICATE_LIB_OK']='True'

#WGAN
def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)

def custom_loss():
    global g1_loss, g2_loss  

    def loss(y_true, y_pred):
        #return (1-alpha) * loss_cls(y1, x1) + alpha * loss_loc(y2, x2)    
        global g1_loss, g2_loss 
        g1_loss = np.float32(g1_loss)
        g2_loss = np.float32(g2_loss)
        f = (g1_loss-g2_loss) + K.mean(y_true-y_pred)*0
        return f 
    return loss

def g1l(score):
    def loss(y_true, y_pred):
        #return (1-alpha) * loss_cls(y1, x1) + alpha * loss_loc(y2, x2)
        global g1_loss, score
        score = np.float32(score)
        #g1_loss = K.mean(y_true-y_pred)
        #return g1_loss
        return score + K.mean(y_true-y_pred)*0
    return loss

def g2l(score):
    def loss(y_true, y_pred):
        #return (1-alpha) * loss_cls(y1, x1) + alpha * loss_loc(y2, x2)
        global g2_loss, score
        score = np.float32(score)*-1
        #g2_loss = K.mean(y_true-y_pred)*-1
        #return g2_loss
        return score + K.mean(y_true-y_pred)*0
    return loss

def custom_loss_gan1(y_true,y1_pred):
    
    return K.mean(K.binary_crossentropy(y_true, y1_pred))*-1
    

def custom_loss_gan2(y_true, y2_pred):
    
    return K.sqrt(K.binary_crossentropy(y_true, y2_pred))

# define the standalone discriminator model
def define_discriminator(n_inputs):
    global g1_loss, g2_loss
    model = Sequential()
    #model.add(Flatten())
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_shape=(n_inputs, 1)))
    #model.add(Dense(35, activation='relu', kernel_initializer='he_uniform', input_shape=(n_inputs, 1)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='linear'))
    # compile model
    opt = RMSprop(lr=0.00005)
    model.compile(loss=custom_loss(), optimizer='sgd', metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim, n_outputs=1):
    global score
    model = Sequential()
    #model.add(Flatten())
    #print(latent_dim)
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_shape=(latent_dim, 1)))
    #model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    #model.add(Flatten())
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss=g2l(score), optimizer='sgd', metrics=['accuracy'])
    return model

##Second generator

def define_generator2(latent_dim, n_outputs=1):
    global score
    model = Sequential()
    #model.add(Flatten())
    #print(latent_dim)
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_shape=(latent_dim, 1)))
    #model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    #model.add(Flatten())
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss=g2l(score), optimizer='sgd', metrics=['accuracy'])
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, generator2, discriminator):
    global g1_loss, g2_loss
    # make weights in the discriminator not trainable
    #discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the second generator
    model.add(generator2)
    model.add(discriminator)
    # compile model
    #model.add(Flatten())
    model.compile(loss=custom_loss(), optimizer='sgd', metrics=['accuracy'])
    
    model.summary()
    return model


def generate_real(generator, x):
    # generate random Gaussian values
    # seed random number generator
    #seed(0)
    x = int (x)
    # generate some Gaussian values
    values = randn(x) - 0.5
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
    #values=generator.predict(values)
    y = ones((x, 1))
    y = y.reshape(1, x, 1)
    #generator.fit(values, y, epochs=10)
    
    #values = generator.predict(values)
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
    #seed(0)
    # generate random Gaussian values
    # seed random number generator
    x = int (x)
    # generate some Gaussian values
    values = randn(x) + 0.5
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
    #values = generator2.predict(values)
    #y = zeros((x, 1))
    y = zeros((x,1))
    y = y.reshape(1, x, 1)
    #generator2.fit(values, y, epochs=10)
    #values = generator2.predict(values)
    
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

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	x_input = randn(latent_dim * n)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n, latent_dim)
	return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator2, x):
    x = int (x)
    values = randn(x)
    y = zeros((x,1))
    y = y.reshape(1, x, 1)
    values = values.reshape(1, x, 1)
    generator2.fit(values,y)
    values = generator2.predict(values)
    
    return values, y

# use the generator to generate n fake examples, with class labels
def generate_real_samples(generator, x):

    x = int (x)
    values = randn(x)
    y = ones((x,1))
    y = y.reshape(1, x, 1)
    values = values.reshape(1, x, 1)
    generator.fit(values,y)
    values = generator.predict(values)
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
    print ("bp", batch_per_epoch)
    global score, g1_loss, g2_loss, dis_loss, yl, latent_dim
    # manually enumerate epochs
    #g1_tmp, g2_tmp = list(), list()
    x_real, y_real = generate_real(g_model, batch_size)
    x_fake, y_fake = generate_fake(g_model2, batch_size)
    x_real= x_real.reshape(1, batch_size,1)
    y_real = y_real.reshape(1, batch_size, 1)
    x_fake = x_fake.reshape(1, batch_size, 1)
    y_fake = y_fake.reshape(1, batch_size, 1)
            #print ("before", x_real, " shape", x_real.shape)
    dxx = np.concatenate([x_real, x_fake], axis=1)
    dyy = np.concatenate([y_real, y_fake], axis=1)
    d_model.fit(dxx,dyy, epochs=30)
    #g_p = d_model.predict(dxx)
            #print ("after", dx, " shape", dx.shape)
    #
            
    #score = d_model.predict(dx)
            
            #g_model.train_on_batch(x_real, y_real)
    
    for i in range(n_epochs):
        # prepare real samples
        for j in range(batch_per_epoch):
            #score_each.reshape(-1,1)
            #print("##################score", score_each.shape)
            # update discriminator
        #d_model.train_on_batch(x_real, y_real)
        #d_model.train_on_batch(x_fake, y_fake)
        # prepare points in latent space as input for the generator
        #x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        #y_gan = ones((n_batch, 1))
       # xx_gan = generate_latent_points(latent_dim, n_batch)
       # yy_gan = ones((n_batch, 0))
        # update the generator via the discriminator's error
            x_real_n, y_real_n = generate_real_samples(g_model, batch_size)

            x_fake_n, y_fake_n = generate_fake_samples(g_model2, batch_size)
            #print ("x_real:,,,,,,,,,,,,,,,,,,,,",x_real_n.shape)
            
            x_real = x_real.reshape(-1,1)
            x_real_n = x_real_n.reshape(-1,1)
            x_fake_n = x_fake_n.reshape(-1,1)
            nn = NearestNeighbors(n_neighbors=1, metric=dis, algorithm='brute').fit(x_real)
            dists1, idxs1 = nn.kneighbors(x_real_n)
            #pdis_r = pdist(x_real, dis)
            id1 = idxs1[:,0]
            n1 = x_real[id1]
            #plt.scatter(dists, dists, marker='*')
            x_fake = x_fake.reshape(-1,1)
            nn = NearestNeighbors(n_neighbors=1, metric=dis, algorithm='brute').fit(x_fake)
            dists2, idxs2 = nn.kneighbors(x_fake_n)
            id2 = idxs2[:,0]
            n2 = x_fake[id2]
            dcn = np.concatenate([x_real_n, x_fake_n], axis=1)
            dx = np.concatenate([n1, n2], axis=1)
            dx = dx.reshape(1,batch_size*2, 1)
            dy = np.concatenate([y_real_n, y_fake_n], axis=1)
            dy = dy.reshape(1,batch_size*2, 1)
            n1 = n1.reshape(1, batch_size,1)
            n2 = n2.reshape(1, batch_size,1)
            x_real_n= x_real_n.reshape(1, batch_size,1)
            x_fake_n= x_fake_n.reshape(1, batch_size,1)
            dcn = dcn.reshape(1, batch_size*2,1)
            print ("Fitting Discriminator...............................................................................")
            d_model.fit(dx, dy)
            score_each = d_model.predict(dx)
            score_concate =  d_model.predict(dxx)
            plt.scatter(x_real, x_real, marker = '^')
            plt.scatter(x_fake, x_fake, marker = 'o')
            plt.scatter(n1, n1, marker='|')
            plt.scatter(n2, n2, marker='*')
            #plt.show()
            plt.close()
            f_nn.append(np.mean(n2))
            r_nn.append(np.mean(n1))
            #score = 
            #print ("score", score, "score", score.shape)
            
            #y1_pred = g_model.predict(x_real)
            #y2_pred = g_model2.predict(x_fake)
            score = np.mean(score_each)         
            g1_loss = gan_model.train_on_batch(n1, y_real)
            g2_loss = gan_model.train_on_batch(n2, y_fake)
            dis_loss = d_model.train_on_batch(dx,dy)
            y_real_n.reshape(1,batch_size,1)
            y_fake_n.reshape(1,batch_size,1)
            #x_real_n = tf.cast(x_real_n, "float32")
            #x_fake_n = tf.cast(x_fake_n, "float32")
            #y_real_n = tf.cast(y_real_n, "float32")
            #y_fake_n = tf.cast(y_fake_n, "float32")
            #gan_loss = gan_model.train_on_batch(x_real_n,y_real_n)
            #gan_loss = gan_model.train_on_batch(x_fake_n,y_fake_n)
            #gan_model.train_on_batch(x_fake_n, y_fake_n)
            #dis_loss = gan_model.evaluate(dx,dy)
            print ("score", np.mean(score), "g1_loss", g1_loss, "g2_loss", g2_loss, "dis_loss", dis_loss)
            #yl.append(score)
            #score = np.mean(yl)
            #d_model.train_on_batch(g1_loss[0], g2_loss[0])
            c1_hist.append(np.mean(dis_loss))
            g1_hist.append(np.mean(g1_loss))
            g2_hist.append(np.mean(g2_loss))
            #d_model.compile(optimizer='sgd',loss=custom_loss(g1_loss,g2_loss))
            #g_model.compile(optimizer='sgd', loss=g1l(score))
            #g_model2.compile(optimizer='sgd', loss=g2l(score))
            final_plot(n1, n2, score_each, x_real, x_fake, dxx, score_concate)
            plot_history(c1_hist,g1_hist,g2_hist)
            
           
        #print("g1: ", g1_loss, "g2: ", g2_loss, "d:", dis_loss)
        # evaluate the model every n_eval epochs
        
        #if (i+1) % n_eval == 0:
           #summarize_performance(i, g_model, g_model2, d_model, size)
        
    #final_plot(r_nn, f_nn, c1_hist)
# size of the latent space
# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, g1_hist, g2_hist):
	# plot history
	pyplot.plot(d1_hist, label='discriminator')
	pyplot.plot(g1_hist, label='generator_real')
	pyplot.plot(g2_hist, label='generator_fake')
	pyplot.legend()
	pyplot.savefig('/Users/masnoonnafees/Documents/GAN/project/Plot/loss{}.png'.format(count), format="PNG")
	#pyplot.show()

def final_plot(n1, n2, score_each, xr, xf, dxx, score_concate):
    #pyplot.scatter(r_nn, c1_hist, color='green')
    global count
    #pyplot.scatter(f_nn, c1_hist, color='blue')
    pyplot.scatter(xr, xr*0, color = 'black')
    pyplot.scatter(xf, xf*0, color = 'red')
    pyplot.scatter(dxx, score_concate, color='orange')
    #pyplot.plot('Nearest Neigbour','discriminator score')
    pyplot.scatter(n1[:,0:250,:], score_each[:,0:250,:], color='green')
    pyplot.scatter(n2[:,0:250,:], score_each[:,250:500,:], color='blue')
    #pyplot.scatter(score_each[:,0:500,:], score_each[:,0:500,:], color='yellow')
    
    #pyplot.xlabel("g1 & g2 loss")
    #pyplot.ylabel("Discriminator Score")
    pyplot.savefig('/Users/masnoonnafees/Documents/GAN/project/Plot/NN{}.png'.format(count), format="PNG")                     
    count = count+1
    #pyplot.show()
    pyplot.close()


size = 2500
batch_size = int(size/10)
dis = 'euclidean'
g1_loss = 0
g2_loss = 0
dis_loss = 0.1
score = 0
yl = []
c1_hist, g1_hist, g2_hist = [], [], []
r_nn, f_nn = [], []
latent_dim = 5
count = 1
count2 = 1
# create the discriminator
discriminator = define_discriminator(batch_size*2)
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