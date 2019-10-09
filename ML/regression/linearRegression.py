# encoding = utf - 8
# /usr/bin/python3

"""The initial parameter of W can influence the iteration steps heavily."""

import numpy as np
import random
import matplotlib.pyplot as plt


num_inputs = 2
num_examples = 1000
true_w = [22.5, -43.3]


features = np.random.rand(num_examples, num_inputs)
labels = features @ np.array(true_w) 
labels += np.random.rand(labels.shape[0]) * 0.01	    # Noise
    

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    for i in range(0, num_examples, batch_size):
        yield features[i: i + batch_size], labels[i: i + batch_size]



batch_size = 5


class linReg():
    def __init__(self, num_inputs):
        self.w = np.array([20., 20.])
        #self.b = np.random.rand(1, )


    def squared_loss(self, y_hat, y):
        squared_err = (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
        res = np.sqrt(np.mean(squared_err))
        return res


    def grad(self, X, y, W = None):
        return np.array([
            np.mean(X[:, 0] * np.mean(self.y_hat(X) - y)),
            np.mean(X[:, 1] * np.mean(self.y_hat(X) - y))
            #2*np.mean(self.y_hat(X) - y)
        ])


    def gd(self, grad, lr):
        self.w -= (lr * grad)
        #self.b -= (lr * grad)[-1]


    def y_hat(self, X):
        return X @ self.w 


    def parameters(self):
        return [self.w]



lr = 0.01
num_epochs = 100

model = linReg(num_inputs)

losses, grads = [], []


for epoch in range(num_epochs):  
    for X, y in data_iter(batch_size, features, labels):
        ### loss
        loss = model.squared_loss(model.y_hat(X), y)
        losses.append(loss)

        ### grad
        grad = model.grad(X, y)
        grads.append(np.mean(grad))

        ### update
        model.gd(grad, lr)



print("True W:", true_w)
print('Predict W:', model.parameters())

### visualization
plt.subplot(121)
plt.plot([i for i in range(len(grads))], grads)
plt.title("Gradient update process")
plt.subplot(122)
plt.plot([i for i in range(len(losses))], losses)
plt.title("Losses")
plt.show()


