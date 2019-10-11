# encoding = utf - 8
# /usr/bin/python3

"""The initial parameter of W can influence the iteration steps heavily."""

import numpy as np
import random
import matplotlib.pyplot as plt
import time
from optimizer import *
import os


num_inputs = 2
num_examples = 1000
true_w = [22.5, -43.3]
true_b = 54.2
features = np.random.rand(num_examples, num_inputs)
labels = features @ np.array(true_w) + true_b
labels += np.random.rand(labels.shape[0]) * 0.01	    # Noise
    

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    for i in range(0, num_examples, batch_size):
        yield features[i: i + batch_size], labels[i: i + batch_size]




def train(num_epochs = 100, batch_size = 5, cliff = 0.5, Optimizer = 'batchGD', \
    Model = linReg, visualization = True, saveImage=True):
    s = time.time()
    lr = 0.05
    # early threashould

    model = Model(num_inputs)
    global losses, grads
    losses, grads = [], []

    for epoch in range(num_epochs):
        #print("epoch :", epoch)
        batchLoss = []
        stopRec = -1
        for X, y in data_iter(batch_size, features, labels):
            ### loss
            loss = model.squared_loss(model.y_hat(X), y)
            batchLoss.append(loss)

            ### grad Optimization
            if Optimizer == 'batchGD':
                grad = model.grad(X, y)
                model.batchGD(grad, lr)
            elif Optimizer == 'SGD':
                rand = random.randint(0, batch_size-1)
                grad = model.grad(np.array([X[rand]]), np.array([y[rand]]))
                model.batchGD(grad, lr)
            elif Optimizer == 'momentum':
                grad = model.grad(X, y)
                model.sgd_momentum(grad, lr)
            elif Optimizer == 'adagrad':
                grad = model.grad(X, y)
                model.sgd_AdaGrad(grad, lr)
            elif Optimizer == 'rmsprop':
                grad = model.grad(X, y)
                model.sgd_RMSProp(grad, lr)
            elif Optimizer == 'adadelta':
                grad = model.grad(X, y)
                model.sgd_AdaDelta(grad)
            elif Optimizer == 'adam':
                grad = model.grad(X, y)
                model.sgd_Adam(grad, lr)
            else:
                raise Exception("Havnt specfy your Optimizer")


            ### update and store
            model.batchGD(grad, lr)
            grads.append(np.mean(grad))

        losses.extend(batchLoss)
        ### early stop
        if np.mean(batchLoss) < cliff: 
            stopRec = epoch
            break


    print(Optimizer, ':')
    print("True W:", true_w, true_b)
    print('Predict W:', model.parameters())
    print("Early stop as epoch: {}".format(stopRec))
    print('time eclapse: ', time.time() - s)

    ### visualization
    if visualization:
        plt.subplot(121)
        plt.plot([i for i in range(len(grads))], grads)
        plt.title("Gradient update process")
        plt.subplot(122)
        plt.plot([i for i in range(len(losses))], losses)
        plt.title("Losses")
        if saveImage:
            restore = os.getcwd()
            os.chdir('../../images/')
            plt.savefig('{}grad&loss.png'.format(Optimizer))
            os.chdir(restore)
        plt.close('all')
        #plt.show()



### bathGradient descent
train(Optimizer="batchGD")

### stochastic gradient descent
train(Optimizer="SGD")

### momentum
train(Optimizer='momentum', Model=momentumLinreg)

### adagrad
train(Optimizer='adagrad', Model=AdaGradLinreg, visualization=True)

### RMSProp
train(Optimizer='rmsprop', Model=RMSPropLinreg, visualization=True)

### AdaDelta
train(Optimizer='adadelta', Model=AdaDeltaLinreg, visualization=True)

### Adam
train(Optimizer='adam', Model=AdamLinreg, visualization=True)




