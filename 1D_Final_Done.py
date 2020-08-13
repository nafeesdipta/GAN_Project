import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from numpy import hstack
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import randn
from numpy.random import randint
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.layers import Layer, Dense, Flatten, BatchNormalization, LeakyReLU, Dropout
from matplotlib import pyplot
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.spatial.distance import pdist
import os
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input
from tensorflow.keras.constraints import Constraint
from statistics import mean

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

g1_loss = []
g2_loss = []
dis_loss = []
score = []


class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return K.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}
# WGAN
def wasserstein_loss(y_true, y_pred): #Here I am multiplying the output(score)
    
    return K.mean(y_true*y_pred)


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
    const = ClipConstraint(0.01)
    model = Sequential()
    # model.add(Flatten())
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    #model.add(Dense(5, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    # compile model
    opt = RMSprop(lr=0.00005)
    validity = model(input_shape_d)
    model = Model(input_shape_d, validity, name='Discriminator')
    #model.compile(loss=wasserstein_loss, optimizer=opt, metrics=['accuracy'])
    return model



# define the standalone generator model
def define_generator(latent_dim, input_shape_g1, n_outputs=1):
    global score
    model = Sequential()
    # model.add(Flatten())
    # print(latent_dim)
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    #model.add(Dense(5, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    # model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    #model.add(Flatten())
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(n_outputs, activation='linear'))
    #gen_real = model(input_shape_g1)
    #model = Model(input_shape_g1, gen_real, name='Generator_1')
    #model.compile(loss=wasserstein_loss, optimizer='adam', metrics=['accuracy'])
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
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(n_outputs, activation='linear'))
    #model.compile(loss=wasserstein_loss, optimizer='adam', metrics=['accuracy'])
    #gen_fake = model(input_shape_g2)
    #model = Model(input_shape_g2, gen_fake, name='Generator_2')
    #model.compile(loss=wasserstein_loss, optimizer='adam', metrics=['accuracy'])
    return model



# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, generator2, discriminator, input_shape_g1, input_shape_g2, input_shape_d):
    global g1_loss, g2_loss
    # make weights in the discriminator not trainable
    #discriminator.trainable = False
    # connect them
    generator.trainable = True
    generator2.trainable = True
    discriminator.trainable = True
    #input_gen1 = Input(shape=input_shape_g)
    #input_gen2 = Input(shape=input_shape_g)
    d_input1 = generator(input_shape_g1)
    d_input2 = generator2(input_shape_g2)
    #d_input = np.concatenate([d_input1, d_input2])
    valid_r = discriminator(d_input1)
    valid_f = discriminator(d_input2)
    model = Model(inputs=[input_shape_g1, input_shape_g2], outputs=[valid_r,valid_f], name='Combined_Advarsarial')
    opt = RMSprop(lr=0.00005)
    #model.compile(loss=[wasserstein_loss, wasserstein_loss], optimizer = opt)
    #model.summary()
    #plot_model(model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)
    return model



def generate_real(generator, x):
    x = int(x)
    # generate some Gaussian values
    #values = randn(x) + 0.5
    values = np.random.uniform(-0.5, 2.0, x)
    y = -ones((x, 1))

    return values, y


def generate_fake(generator2, x):
    x = int(x)
    # generate some Gaussian values
    #values = randn(x) - 0.5
    values = np.random.uniform(-2.0, 0.5, x)
    y = ones((x, 1))

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
def generate_fake_samples(generator2, latent_dim, n, dataset):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    #ix = randint(0, dataset.shape[0], n)
    #x_input = dataset[ix]
    # predict outputs
    #print("xxxxxxxbefore", x_input.shape)
    X = generator2.predict(x_input)
    # create class labels
    y = ones((n, 1))
    #print("xxxxxxx", x_input.shape)

    return X, y


def generate_real_samples(generator, latent_dim, n, dataset):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    #ix = randint(0, dataset.shape[0], n)
    #x_input = dataset[ix]
    X = generator.predict(x_input)
    # create class labels
    y = -ones((n, 1))
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


def step(X, y, model):

    with tf.GradientTape() as tape:
        pred = model(X)
        loss = wasserstein_loss(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)

    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss
        
def generator_loss1(fake_output):
    return wasserstein_loss(tf.ones_like(fake_output)*-1, fake_output)

def generator_loss2(fake_output):
    return wasserstein_loss(tf.ones_like(fake_output), fake_output)
# train the generator and discriminator
def train(g_model, g_model2, d_model, gan_model, batch_size, size, n_epochs=150, n_eval=2):
    # determine half the size of one batch, for updating the discriminator
    # batch_size = 100
    batch_size = 32
    batch_per_epoch = int(size / batch_size)
    print("bp", batch_per_epoch)
    global score, g1_loss, g2_loss, dis_loss, yl, latent_dim
    # manually enumerate epochs
    # g1_tmp, g2_tmp = list(), list()
    x_real, y_real = generate_real(g_model, size)
    x_fake, y_fake = generate_fake(g_model2, size)
    # print ("before", x_real, " shape", x_real.shape)
    dxx = np.concatenate([x_real, x_fake])
    dyy = np.concatenate([y_real, y_fake])
    opt = RMSprop(lr=0.0005)
    xx = x_real
    yy = x_fake
    writer = tf.summary.create_file_writer('Graph_2')
    for i in range(n_epochs):
        for j in range(batch_per_epoch):
            
            x_real_n, y_real_n = generate_real_samples(g_model, latent_dim, batch_size, dxx)

            x_fake_n, y_fake_n = generate_fake_samples(g_model2, latent_dim, batch_size, dxx)
            #ix = randint(0, x_real.shape[0], batch_size)
            #x_real = x_real[ix]
            #x_fake = x_fake[ix]
            x_real = x_real.reshape(-1, 1)
            x_real_n = x_real_n.reshape(-1, 1)
            nn = NearestNeighbors(n_neighbors=1, metric=dis, algorithm='brute').fit(x_real)
            dists1, idxs1 = nn.kneighbors(x_real_n)
            id1 = idxs1[:, 0]
            n1 = x_real[id1]

            x_fake_n = x_fake_n.reshape(-1, 1)
            x_fake = x_fake.reshape(-1, 1)
            nn = NearestNeighbors(n_neighbors=1, metric=dis, algorithm='brute').fit(x_fake)
            dists2, idxs2 = nn.kneighbors(x_fake_n)
            id2 = idxs2[:, 0]
            n2 = x_fake[id2]
            dcn = np.concatenate([x_real_n, x_fake_n])
            dyb = np.concatenate([y_real_n, y_fake_n])
            dx = np.concatenate([n1, n2])
            dx = dx.reshape(-1, 1)
            dxx = dxx.reshape(-1,1)
            dcn = dcn.reshape(-1,1)
            n1 = n1.reshape(-1,1)
            n2 = n2.reshape(-1,1)
            x_real_n = x_real_n.reshape(-1,1)
            y_real_n = y_real_n.reshape(-1,1)
            x_fake_n = x_fake_n.reshape(-1,1)
            y_fake_n = y_fake_n.reshape(-1,1)
            g1 = d_model.predict(x_real_n)
            g2 =  d_model.predict(x_fake_n)
            #step(dxx, dyy, g_model, g_model2, d_model,3)
            #d_model.train_on_batch(dxx, dyy)
            score_each = d_model.predict(dx)
            score_concate = d_model.predict(dxx)
            score = np.mean(score_each)
            score_real = d_model.predict(x_real)
            score_fake = d_model.predict(x_fake)
            #with tf.GradientTape() as tape:
            #    pred = d_model(n1, training=True)
            #    loss = wasserstein_loss(y_real_n, pred)
            #grads = tape.gradient(loss, d_model.trainable_variables)
            #opt.apply_gradients(zip(grads, d_model.trainable_variables))
            #d1_loss = loss
            ycn = ones((batch_size*2, 1))
            #with tf.GradientTape() as tape:
            #    pred = d_model(n2, training=True)
            #    loss = wasserstein_loss(y_fake_n, pred)
            #grads = tape.gradient(loss, d_model.trainable_variables)
            #opt.apply_gradients(zip(grads, d_model.trainable_variables))
            #d2_loss = loss
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
            y_real_n = -ones((batch_size,1))
            #y_real_n = np.concatenate([y_real_n,y_real_n])
            y_fake_n = ones((batch_size,1))
            #y_fake_n = np.concatenate([y_fake_n,y_fake_n])
            y_real_n = y_real_n.reshape(-1, latent_dim)
            y_fake_n = y_fake_n.reshape(-1, latent_dim)

            #with tf.GradientTape() as tape:
            #    grads = tape.gradient(dis_loss, d_model.trainable_weights)
            #    opt.apply_gradients(zip(grads, d_model.trainable_weights))
            #with writer.as_default():
            #    for j in grads:
            #        tf.summary.histogram('Discriminators_grad', j, step=count)
            #    tf.summary.histogram('Discriminators_loss', dis_loss, step=count)
            
            #with tf.GradientTape() as tape:
            #    #pred =  d_model(x_gan_r)
            #    pred = g_model(n1, training=True)

             #   loss = wasserstein_loss(y_fake_n, pred)
              #  grads = tape.gradient(loss, g_model.trainable_variables)
               # opt.apply_gradients(zip(grads, g_model.trainable_variables))
            
            #g1_loss = loss
            #with writer.as_default():
             #   for j in grads:
              #      tf.summary.histogram('Generator1_grad', j, step=count)
               # tf.summary.histogram('Generator1_loss', g1_loss, step=count)


            #with tf.GradientTape() as tape:
             #   #pred = d_model(x_gan_f)
              #  pred = g_model2(x_gan_f, training=True)
               # loss = wasserstein_loss(y_real_n, pred)
            #grads = tape.gradient(loss, g_model2.trainable_variables)
            #opt.apply_gradients(zip(grads, g_model2.trainable_variables))
            #g2_loss = loss
            #with writer.as_default():
             #   for j in grads:
              #      tf.summary.histogram('Generator2_grad', j, step=count)
               # tf.summary.histogram('Generator2_loss', g2_loss, step=count)
            opt = RMSprop(lr=0.00005)
            for a in range(5):
                with tf.GradientTape() as tape:
                #pred =  d_model(x_gan_r)
                    pred = d_model(n1, training=True)
                    loss = wasserstein_loss(y_real_n, pred)
            
                grads = tape.gradient(loss, d_model.trainable_variables)
                opt.apply_gradients(zip(grads, d_model.trainable_variables))
            
                with tf.GradientTape() as tape:
                #pred =  d_model(x_gan_r)
                    pred = d_model(n2, training=True)
                    loss = wasserstein_loss(y_fake_n, pred)
            
                grads = tape.gradient(loss, d_model.trainable_variables)
                opt.apply_gradients(zip(grads, d_model.trainable_variables))

            opt = RMSprop(lr=0.00001)
            with tf.GradientTape() as tape:
                pred1, pred2 = gan_model([x_gan_r,x_gan_f])
                loss1 = wasserstein_loss(y_fake_n, pred1)
                loss2 = wasserstein_loss(y_real_n, pred2)
            grads = tape.gradient([loss1,loss2], gan_model.trainable_variables)
            opt.apply_gradients(zip(grads, gan_model.trainable_variables))
            
            
            dis_update = tf.math.subtract(pred1,pred2)
            dis_update = tf.multiply(dis_update,-1) 


                #with tf.GradientTape() as tape:
                #    grads = tape.gradient(d2_loss, g_model2.trainable_variables)
                #    opt.apply_gradients(zip(grads, g_model2.trainable_variables))
             
            global count
            g1_loss = loss1
            g2_loss = loss2
            dis_loss = loss
            f_nn.append(np.mean(n2))
            r_nn.append(np.mean(n1))
            print("Score", np.mean(score), "g1_loss", g1_loss, "g2_loss", g2_loss, "dis_loss", dis_loss)
            #print("acc_real", acc_real, "acc_fake", acc_fake)
            #print("//////////////////////////////////////")
            #print (n1)
            #print("########################")
            #print (x_real[0:32])
            #x_real = xx
            #x_fake = yy
            c1_hist.append(np.mean(dis_loss))
            g1_hist.append(np.mean(g1_loss))
            g2_hist.append(np.mean(g2_loss))
            #n1_b,n2_b,score_concate_b,score_each_b = n1,n2,score_concate,score_each
            if count ==1:
                n1_b=n1
                n2_b=n2
                score_concate_b=score_concate
                score_each_b=score_each
                dxx_b = dxx
                x_real_n_b = x_real_n
                x_fake_n_b = x_fake_n
                g1_b = g1
                g2_b = g2
                score_real_b = score_real
                score_fake_b = score_fake
            if count%50 == 0: 
                final_plot(n1, n2, score_each,x_real, x_fake, dxx, score_concate, x_real_n, x_fake_n, g1, g2, n1_b,n2_b,score_each_b,score_concate_b,
                 dxx_b, g1_b,g2_b,x_real_n_b,x_fake_n_b, score_real, score_real_b,score_fake,score_fake_b)
                n1_b=n1
                n2_b=n2
                score_concate_b=score_concate
                score_each_b=score_each
                dxx_b = dxx
                x_real_n_b = x_real_n
                x_fake_n_b = x_fake_n
                g1_b = g1
                g2_b = g2
            count = count + 1
            #plot_history(c1_hist, g1_hist, g2_hist)
            
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

def final_plot(n1, n2, score_each, xr, xf, dxx, score_concate, x_real_n, x_fake_n, g1,g2, n1_b, n2_b, score_each_b, score_concate_b,
 dxx_b, g1_b,g2_b,x_real_n_b,x_fake_n_b, score_real, score_real_b,score_fake,score_fake_b):
    # pyplot.scatter(r_nn, c1_hist, color='green')


    fig2, gg1 = plt.subplots(2)
    # pyplot.plot('Nearest Neigbour','discriminator score')
    gg1[0].scatter(xr, (xr * 0)+2, color='black', label='Ground Value of the user') 
    gg1[0].scatter(xf, (xf * 0)+1, color='red', label = 'Ground value of the non user')
    gg1[0].scatter(dxx_b, score_concate_b, color='orange', label='Dis: Max(D(G(user))-D(G(non user)))')
    gg1[0].scatter(n1_b[0:32, :], score_each_b[0:32, :], color='green', label='G1 minimize user')
    gg1[0].scatter(n2_b[0:32, :], score_each_b[32:64, :], color='blue', label = 'G2 maximize not user')
    gg1[0].scatter(x_real_n_b[0:32, :], g1_b[0:32,:]+4, color='black', label = 'Generated points for user')
    gg1[0].scatter(x_fake_n_b[0:32, :], g2_b[0:32, :]+3, color='red', label = 'Generated points for non user')
    #
    gg1[1].scatter(x_real_n[0:32, :], g1[0:32,:]+4, color='black', label = 'Generated points for user')
    gg1[1].scatter(x_fake_n_b[0:32, :], g2_b[0:32, :]+3, color='red', label = 'Generated points for non user')
    gg1[1].scatter(xr, (xr * 0)+2, color='black', label='Ground Value of the user') 
    gg1[1].scatter(xf, (xf * 0)+1, color='red', label = 'Ground value of the non user')
    gg1[1].scatter(dxx[0:1500], score_concate[0:1500], color='orange', label='Dis: Max(D(G(user))-D(G(non user)))')
    gg1[1].scatter(dxx_b[1500:3000], score_concate_b[1500:3000], color='orange')
    gg1[1].scatter(n1[0:32, :], score_each[0:32, :], color='green', label='G1 minimize user')
    gg1[1].scatter(n2_b[0:32, :], score_each_b[32:64, :], color='blue', label = 'G2 maximize not user')
    pyplot.legend()
    pyplot.savefig("C:\\Users\\nafee\\OneDrive\\Documents\\Plot\\Generator (1) {}.png".format(count), format="PNG")
    pyplot.close()

    fig3, gg2 = plt.subplots(2)
    gg2[0].scatter(xr, (xr * 0)+2, color='black', label='Ground Value of the user') 
    gg2[0].scatter(xf, (xf * 0)+1, color='red', label = 'Ground value of the non user')
    gg2[0].scatter(dxx_b, score_concate_b, color='orange', label='Dis: Max(D(G(user))-D(G(non user)))')
    gg2[0].scatter(n1_b[0:32, :], score_each_b[0:32, :], color='green', label='G1 minimize user')
    gg2[0].scatter(n2_b[0:32, :], score_each_b[32:64, :], color='blue', label = 'G2 maximize not user')
    gg2[0].scatter(x_real_n_b[0:32, :], g1_b[0:32,:]+4, color='black', label = 'Generated points for user')
    gg2[0].scatter(x_fake_n_b[0:32, :], g2_b[0:32, :]+3, color='red', label = 'Generated points for non user')
    #
    gg2[1].scatter(x_real_n_b[0:32, :], g1_b[0:32,:]+4, color='black', label = 'Generated points for user')
    gg2[1].scatter(x_fake_n[0:32, :], g2[0:32, :]+3, color='red', label = 'Generated points for non user')
    gg2[1].scatter(dxx_b[0:1500], score_concate_b[0:1500], color='orange', label='Dis: Max(D(G(user))-D(G(non user)))')
    gg2[1].scatter(dxx[1500:3000], score_concate[1500:3000], color='orange')
    gg2[1].scatter(n2[0:32, :], score_each[32:64, :], color='blue', label = 'G2 maximize not user')
    gg2[1].scatter(n1_b[0:32, :], score_each_b[0:32, :], color='green', label='G1 minimize user') 
    gg2[1].scatter(xr, (xr * 0)+2, color='black', label='Ground Value of the user') 
    gg2[1].scatter(xf, (xf * 0)+1, color='red', label = 'Ground value of the non user')
    pyplot.legend()
    pyplot.savefig("C:\\Users\\nafee\\OneDrive\\Documents\\Plot\\Generator (2) {}.png".format(count), format="PNG")
    # pyplot.xlabel("g1 & g2 loss")
    # pyplot.ylabel("Discriminator Score")
    pyplot.close()

    fig1, dis = plt.subplots(2)
    dis[0].scatter(xr, (xr * 0)+2, color='black', label='Ground Value of the user') 
    dis[0].scatter(xf, (xf * 0)+1, color='red', label = 'Ground value of the non user')
    dis[0].scatter(dxx_b, score_concate_b, color='orange', label='Dis: Max(D(G(user))-D(G(non user)))')
    dis[0].scatter(n2_b[0:32, :], score_each_b[32:64, :], color='blue', label = 'G2 maximize not user')
    dis[0].scatter(n1_b[0:32, :], score_each_b[0:32, :], color='green', label='G1 minimize user')
    dis[0].scatter(x_real_n_b[0:32, :], g1_b[0:32,:]+4, color='black', label = 'Generated points for user')
    dis[0].scatter(x_fake_n_b[0:32, :], g2_b[0:32, :]+3, color='red', label = 'Generated points for non user')
    #
    dis[1].scatter(x_real_n[0:32, :], g1[0:32,:]+4, color='black', label = 'Generated points for user')
    dis[1].scatter(x_fake_n[0:32, :], g2[0:32, :]+3, color='red', label = 'Generated points for non user')
    dis[1].scatter(dxx, score_concate, color='orange', label='Dis: Max(D(G(user))-D(G(non user)))')
    dis[1].scatter(xr, (xr * 0)+2, color='black', label='Ground Value of the user') 
    dis[1].scatter(xf, (xf * 0)+1, color='red', label = 'Ground value of the non user')
    dis[1].scatter(n2[0:32, :], score_each[32:64, :], color='blue', label = 'G2 maximize not user')
    dis[1].scatter(n1[0:32, :], score_each[0:32, :], color='green', label='G1 minimize user')
    pyplot.legend()
    pyplot.savefig("C:\\Users\\nafee\\OneDrive\\Documents\\Plot\\Discriminator{}.png".format(count), format="PNG")
    pyplot.close()

    fig4, axs = plt.subplots(2)
    axs[0].scatter(xr, (xr * 0)+2, color='black', label='Ground Value of the user') 
    axs[0].scatter(xf, (xf * 0)+1, color='red', label = 'Ground value of the non user')
    axs[0].scatter(dxx, score_concate_b, color='orange', label='Dis: Max(D(G(user))-D(G(non user)))')
    # pyplot.plot('Nearest Neigbour','discriminator score')
    axs[0].scatter(n1_b[0:32, :], score_each_b[0:32, :], color='green', label='G1 minimize user')
    axs[0].scatter(n2_b[0:32, :], score_each_b[32:64, :], color='blue', label = 'G2 maximize not user')
    #pyplot.scatter(x_real_n[0:32, :], g1[0:32,:]+4, color='black', label = 'Generated points for G1')
    #pyplot.scatter(x_fake_n[0:32, :], g2[0:32, :]+3, color='red', label = 'Generated points for G2')
    # pyplot.scatter(score_each[:,0:500,:], score_each[:,0:500,:], color='yellow')
    axs[1].scatter(xr, (xr * 0)+2, color='black', label='Ground Value of the user') 
    axs[1].scatter(xf, (xf * 0)+1, color='red', label = 'Ground value of the non user')
    axs[1].scatter(dxx, score_concate, color='orange', label='Dis: Max(D(G(user))-D(G(non user)))')
    # pyplot.plot('Nearest Neigbour','discriminator score')
    axs[1].scatter(n1[0:32, :], score_each[0:32, :], color='green', label='G1 minimize user')
    axs[1].scatter(n2[0:32, :], score_each[32:64, :], color='blue', label = 'G2 maximize not user')
    # pyplot.xlabel("g1 & g2 loss")
    # pyplot.ylabel("Discriminator Score")
    pyplot.legend()
    pyplot.savefig("C:\\Users\\nafee\\OneDrive\\Documents\\Plot\\NN{}.png".format(count), format="PNG")
    #count = count + 1
    # pyplot.show()
    pyplot.close()


size = 1500
latent_dim = 1
batch_size = int(size / 10)
dis = 'euclidean'
yl = []
c1_hist, g1_hist, g2_hist = [], [], []
r_nn, f_nn = [], []
count = 1
count2 = 1
# create the discriminator
input_shape_d = Input(shape=(latent_dim,))
input_shape_g1 = Input(shape=(latent_dim,))
input_shape_g2 = Input(shape=(latent_dim,))
# create the generator
generator = define_generator(latent_dim, input_shape_g1)
generator2 = define_generator2(latent_dim, input_shape_g2)
discriminator = define_discriminator(input_shape_d)
# create the gan
gan_model = define_gan(generator, generator2, discriminator, input_shape_g1, input_shape_g2, input_shape_d)
# train model
#gan_model.summary()
# plot gan model
#plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)
train(generator, generator2, discriminator, gan_model, batch_size, size)
print(c1_hist.count)
print(len(c1_hist))