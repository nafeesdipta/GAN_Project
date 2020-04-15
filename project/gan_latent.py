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
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.layers import Layer, Dense, Flatten, BatchNormalization, LeakyReLU, Dropout
from matplotlib import pyplot
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.spatial.distance import pdist
import os
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input
from statistics import mean

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

g1_loss = []
g2_loss = []
dis_loss = []
score = []

# WGAN
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def wasserstein_loss2(y_true, y_pred):
    return K.mean(y_true * y_pred)*-1

def custom_loss(g1_loss,g2_loss):
    #global g1_loss, g2_loss

    def loss(y_true, y_pred):
        # return (1-alpha) * loss_cls(y1, x1) + alpha * loss_loc(y2, x2)
        global g1_loss, g2_loss
        g1_loss = np.float32(g1_loss)
        g2_loss = np.float32(g2_loss)
        f = (g1_loss - g2_loss) + K.mean(y_true - y_pred) * 0
        return f

    return loss


def g1l(score):
    def loss(y_true, y_pred):
        # return (1-alpha) * loss_cls(y1, x1) + alpha * loss_loc(y2, x2)
        global g1_loss, score
        score = np.float32(score)
        # g1_loss = K.mean(y_true-y_pred)
        # return g1_loss
        return score + K.mean(y_true - y_pred) * 0

    return loss


def g2l(score):
    def loss(y_true, y_pred):
        # return (1-alpha) * loss_cls(y1, x1) + alpha * loss_loc(y2, x2)
        global g2_loss, score
        score = np.float32(score*-1)
        # g2_loss = K.mean(y_true-y_pred)*-1
        # return g2_loss
        return score + K.mean(y_true - y_pred) * 0

    return loss


def custom_loss_gan1(y_true, y1_pred):
    return K.mean(K.binary_crossentropy(y_true, y1_pred)) * -1


def custom_loss_gan2(y_true, y2_pred):
    return K.sqrt(K.binary_crossentropy(y_true, y2_pred))


# define the standalone discriminator model
def define_discriminator(input_shape_d, n_inputs=1):
    global g1_loss, g2_loss
    model = Sequential()
    # model.add(Flatten())
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    # model.add(Dense(35, activation='relu', kernel_initializer='he_uniform', input_shape=(n_inputs, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    # compile model
    opt = RMSprop(lr=0.0005)
    validity = model(input_shape_d)
    model = Model(input_shape_d, validity, name='Discriminator')
    model.compile(loss=wasserstein_loss, optimizer='adam', metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim, input_shape_g1, n_outputs=1):
    global score
    model = Sequential()
    # model.add(Flatten())
    # print(latent_dim)
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    # model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    #model.add(Flatten())
    model.add(Dense(n_outputs, activation='linear'))
    #
    gen_real = model(input_shape_g1)
    model = Model(input_shape_g1, gen_real, name='Generator_1')
    model.compile(loss=wasserstein_loss, optimizer='adam', metrics=['accuracy'])
    return model


##Second generator

def define_generator2(latent_dim, input_shape_g2, n_outputs=1):
    global score
    model = Sequential()
    # model.add(Flatten())
    # print(latent_dim)
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    # model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    #model.add(Flatten())
    model.add(Dense(n_outputs, activation='linear'))
    #model.compile(loss=wasserstein_loss, optimizer='adam', metrics=['accuracy'])
    gen_fake = model(input_shape_g2)
    model = Model(input_shape_g2, gen_fake, name='Generator_2')
    model.compile(loss=wasserstein_loss, optimizer='adam', metrics=['accuracy'])
    return model



# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, generator2, discriminator, input_shape_g1, input_shape_g2, input_shape_d):
    global g1_loss, g2_loss
    # make weights in the discriminator not trainable
    #discriminator.trainable = False
    # connect them
    generator.trainable = True
    generator2.trainable = True
    discriminator.trainable = False
    #input_gen1 = Input(shape=input_shape_g)
    #input_gen2 = Input(shape=input_shape_g)
    d_input1 = generator(input_shape_g1)
    d_input2 = generator2(input_shape_g2)
    #d_input = np.concatenate([d_input1, d_input2])
    valid_r = discriminator(d_input1)
    valid_f = discriminator(d_input2)
    model = Model(inputs=[input_shape_g1, input_shape_g2], outputs=[valid_r,valid_f], name='Combined_Advarsarial')
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=[wasserstein_loss, wasserstein_loss], opttimizer = opt)
    return model



def generate_real(generator, x):
    x = int(x)
    # generate some Gaussian values
    values = randn(x) - 2.5

    y = ones((x, 1))

    return values, y


def generate_fake(generator2, x):
    x = int(x)
    # generate some Gaussian values
    values = randn(x) + 2.5

    y = zeros((x, 1))

    return values, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
    # generate points in the latent space
    x_input = randn(latent_dim * n)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    #print("xxxxxxx",x_input)

    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    #print("xxxxxxxbefore", x_input.shape)
    X = generator.predict(x_input)
    # create class labels
    y = zeros((n, 1))
    #print("xxxxxxx", x_input.shape)

    return X, y


def generate_real_samples(generator, latent_dim, n):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = ones((n, 1))
    return X, y


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


def step(X, y, g_model, g_model2, d_model, num):
    # keep track of our gradients
    global g1_loss, g2_loss, score
    #score = d_model.predict(X)
    opt = RMSprop(lr=0.0005)
    if num == 1:
        model = g_model
        l = g1l
    elif num == 2:
        model = g_model2
        l = g2l
    elif num == 3:
        model = d_model
        l = custom_loss

    with tf.GradientTape() as tape:
        # make a prediction using the model and then calculate the
        # loss
        pred = model(X)
        loss = binary_crossentropy(y, pred)

        ##pred = model(X)
        # Get the loss
        # replace loss with our score
        # calculate the gradients using our tape and then update the
        # model weights
        grads = tape.gradient(loss, model.trainable_variables)
        #I was facing Gradiant NUll issues, so if grad == null, I am replacing with zero like tensor
        #grads = [grad if grad is not None else tf.zeros_like(var)
        #        for var, grad in zip(model.trainable_variables, grads)]
        opt.apply_gradients(zip(grads, model.trainable_variables))
        #loss = tf.convert_to_tensor(l, np.float32)
        


# train the generator and discriminator
def train(g_model, g_model2, d_model, gan_model, batch_size, size, n_epochs=100, n_eval=2):
    # determine half the size of one batch, for updating the discriminator
    # batch_size = 100
    batch_per_epoch = int(size / batch_size)
    print("bp", batch_per_epoch)
    global score, g1_loss, g2_loss, dis_loss, yl, latent_dim
    # manually enumerate epochs
    # g1_tmp, g2_tmp = list(), list()
    x_real, y_real = generate_real(g_model, batch_size)
    x_fake, y_fake = generate_fake(g_model2, batch_size)
    # print ("before", x_real, " shape", x_real.shape)
    dxx = np.concatenate([x_real, x_fake])
    dyy = np.concatenate([y_real, y_fake])
    #print("xreal shape...............", dxx.shape)
    #X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(dxx, dyy, test_size=0.3, train_size=0.7)
    #X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    #hist = d_model.fit(X_train, Y_train,
    #      batch_size=32, epochs=10,
    #      validation_data=(X_val, Y_val))
    #d_model.fit(dxx, dyy, epochs=10)
    for i in range(n_epochs):
        for j in range(batch_per_epoch):
            
            x_real_n, y_real_n = generate_real_samples(g_model, latent_dim, batch_size)

            x_fake_n, y_fake_n = generate_fake_samples(g_model2, latent_dim, batch_size)

            x_real = x_real.reshape(1, -1)
            x_real_n = x_real_n.reshape(1, -1)
            x_fake_n = x_fake_n.reshape(1, -1)
            nn = NearestNeighbors(n_neighbors=1, metric=dis, algorithm='brute').fit(x_real)
            dists1, idxs1 = nn.kneighbors(x_real_n)
            id1 = idxs1[:, 0]
            n1 = x_real[id1]
            x_fake = x_fake.reshape(1, -1)
            nn = NearestNeighbors(n_neighbors=1, metric=dis, algorithm='brute').fit(x_fake)
            dists2, idxs2 = nn.kneighbors(x_fake_n)
            id2 = idxs2[:, 0]
            n2 = x_fake[id2]
            dcn = np.concatenate([x_real_n, x_fake_n])
            dx = np.concatenate([n1, n2])
            dx = dx.reshape(-1, 1)
            dxx = dxx.reshape(-1,1)
            dcn = dcn.reshape(-1,1)
            #step(dxx, dyy, g_model, g_model2, d_model,3)
            #d_model.train_on_batch(dxx, dyy)
            score_each = d_model.predict(dx)
            score_concate = d_model.predict(dxx)
            score = np.mean(score_each)
            x_gan_r = generate_latent_points(latent_dim, batch_size)
            x_gan_f = generate_latent_points(latent_dim, batch_size)
            x_gan_r = x_gan_r.reshape(-1, latent_dim)
            x_gan_f = x_gan_f.reshape(-1, latent_dim)
            x_real_n = x_real_n.reshape(-1, latent_dim)
            #print('x_real_n', x_real_n.shape)
            y_real_n = y_real_n.reshape(-1, latent_dim)
            #g_model.fit(x_real_n, y_real_n)
            x_fake_n = x_fake_n.reshape(-1, latent_dim)
            y_fake_n = y_fake_n.reshape(-1, latent_dim)
            n1 = n1.reshape(-1, latent_dim)
            n2 = n2.reshape(-1, latent_dim)
            #step(x_real_n, y_real_n, g_model, g_model2, d_model, 1)
            #step(x_fake_n, y_fake_n, g_model, g_model2, d_model, 2)
            #g1_loss = g_model.train_on_batch(x_real_n,y_real_n)
            #g2_loss = g_model2.train_on_batch(x_fake_n, y_fake_n)
            #g_model.train_on_batch(x_gan_r, y_real_n)
            #g_model2.train_on_batch(x_gan_r, y_real_n)
            #gan_model.train_on_batch(x_fake_n, y_fake_n)
            #g2_loss = gan_model.evaluate([x_fake_n, y_real_n], []verbose=0)
            #step(dxx, dyy, g_model, g_model2, d_model, 3)
            dis_loss = d_model.train_on_batch(dxx, dyy)
            gan_loss = gan_model.train_on_batch([x_gan_r, x_gan_f],[y_real_n, y_fake_n])
            g1_loss = gan_loss[0]
            g2_loss = gan_loss[1]
            f_nn.append(np.mean(n2))
            r_nn.append(np.mean(n1))
            print("score", np.mean(score), "g1_loss", g1_loss, "g2_loss", g2_loss, "dis_loss", dis_loss, "gan_loss", gan_loss)
            c1_hist.append(np.mean(dis_loss))
            g1_hist.append(np.mean(g1_loss))
            g2_hist.append(np.mean(g2_loss))
            
            final_plot(n1, n2, score_each, x_real, x_fake, dxx, score_concate)
            plot_history(c1_hist, g1_hist, g2_hist)
            
        # print("g1: ", g1_loss, "g2: ", g2_loss, "d:", dis_loss)
        # evaluate the model every n_eval epochs

        # if (i+1) % n_eval == 0:
        # summarize_performance(i, g_model, g_model2, d_model, size)

    # final_plot(r_nn, f_nn, c1_hist)


# size of the latent space
# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, g1_hist, g2_hist):
    # plot history
    pyplot.plot(d1_hist, label='discriminator')
    pyplot.plot(g1_hist, label='generator_real')
    pyplot.plot(g2_hist, label='generator_fake')
    pyplot.legend()
    pyplot.savefig('/Users/masnoonnafees/Documents/GAN/project/Plot/loss{}.png'.format(count), format="PNG")
    pyplot.close()


# pyplot.show()

def final_plot(n1, n2, score_each, xr, xf, dxx, score_concate):
    # pyplot.scatter(r_nn, c1_hist, color='green')
    global count
    # pyplot.scatter(f_nn, c1_hist, color='blue')
    pyplot.scatter(xr, xr * 0, color='black')
    pyplot.scatter(xf, xf * 0, color='red')
    pyplot.scatter(dxx, score_concate, color='orange')
    # pyplot.plot('Nearest Neigbour','discriminator score')
    pyplot.scatter(n1[0:250, :], score_each[0:250, :], color='green')
    pyplot.scatter(n2[0:250, :], score_each[250:500, :], color='blue')
    # pyplot.scatter(score_each[:,0:500,:], score_each[:,0:500,:], color='yellow')

    # pyplot.xlabel("g1 & g2 loss")
    # pyplot.ylabel("Discriminator Score")
    pyplot.savefig('/Users/masnoonnafees/Documents/GAN/project/Plot/NN{}.png'.format(count), format="PNG")
    count = count + 1
    # pyplot.show()
    pyplot.close()


size = 2500
latent_dim = 1
batch_size = int(size / 10)
dis = 'euclidean'
yl = []
c1_hist, g1_hist, g2_hist = [], [], []
r_nn, f_nn = [], []
count = 1
count2 = 1
# create the discriminator
input_shape_d = Input(shape=(1,))
input_shape_g1 = Input(shape=(latent_dim,))
input_shape_g2 = Input(shape=(latent_dim,))
# create the generator
generator = define_generator(latent_dim, input_shape_g1)
generator2 = define_generator2(latent_dim, input_shape_g2)
discriminator = define_discriminator(input_shape_d)
# create the gan
gan_model = define_gan(generator, generator2, discriminator, input_shape_g1, input_shape_g2, input_shape_d)
# train model
gan_model.summary()
# plot gan model
plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)
train(generator, generator2, discriminator, gan_model, batch_size, size)
print(c1_hist.count)
print(len(c1_hist))