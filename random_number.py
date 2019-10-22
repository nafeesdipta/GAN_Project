import numpy as np 
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from numpy import hstack
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import randn

def generate_real(x):
    # generate random Gaussian values
    # seed random number generator
    seed(5)
    # generate some Gaussian values
    values = randn(x) - 0.5
    #y = ones((x, 1))
    values = values.reshape(x, 1)
    #print (values.shape)
    #y = y.reshape(x, 1)
    b = np.hstack((values, np.ones((values.shape[0], 1), dtype= object)))
    #ap = hstack(values, y)
    b = b.reshape(x, 2)
    b[:,b.shape[1]-1] = int(1)
    np.savetxt('real_1d.csv', b, delimiter=',', fmt="%s")
    
    #print(values)
    return values

def generate_fake(x):
    # generate random Gaussian values
    # seed random number generator
    seed(10)
    # generate some Gaussian values
    values = randn(x) + 0.5
    #y = []
    
    values = values.reshape(x, 1)
    print (values.shape)
    #y = y.reshape(x, 1)
    b = np.hstack((values, np.zeros((values.shape[0], 1), dtype =object)))
    b = b.reshape(x, 2)
    b[:,b.shape[1]-1] = int(0)
    #ap = hstack(values, y)
    np.savetxt('fake_1d.csv', b, delimiter=',', fmt="%s")
    #print (ap)
    
    return values


def plot_values(size):
    
    x = generate_fake(size)
    y = generate_real(size)

    plt.scatter(generate_fake(size), generate_fake(size), marker = '^')
    plt.scatter(generate_real(size), generate_real(size), marker= 'o')
    plt.show()


size = 10

plot_values(size)
