# encoding=utf-8
# /usr/bin/python3
import numpy as np


"""vanilla linear regression with gradient descent"""
class linReg():
    def __init__(self, num_inputs):
        self.w = np.random.rand(num_inputs, )
        self.b = np.random.rand(1, )


    def squared_loss(self, y_hat, y):
        squared_err = (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
        res = np.sqrt(np.mean(squared_err))
        return res


    def grad(self, X, y, W = None):
        return np.array([
            np.mean(X[:, 0] * np.mean(self.y_hat(X) - y)),
            np.mean(X[:, 1] * np.mean(self.y_hat(X) - y)),
            1 * np.mean(self.y_hat(X) - y)
        ])


    def batchGD(self, grad, lr):
        self.w -= (lr * grad)[:2]
        self.b -= (lr * grad)[-1]


    def y_hat(self, X):
        return X @ self.w + self.b


    def parameters(self):
        return [self.w, self.b]


"""linear regression with momentum"""
class momentumLinreg(linReg):
    def __init__(self, num_inputs):
        super(momentumLinreg, self).__init__(num_inputs)
        self.wv = np.random.rand(num_inputs, )
        self.bv = np.random.rand(1, )
        self.momentum = 0.5


    def sgd_momentum(self, grad, lr):
        # update momentum v
        self.wv = self.wv * self.momentum + lr * grad[:2]
        self.bv = self.bv * self.momentum + lr * grad[-1]
        # update parameters
        self.w -= self.wv
        self.b -= self.bv


""" adagrad enable the param update with different learning rate """
class AdaGradLinreg(linReg):
    def __init__(self, num_inputs):
        super(AdaGradLinreg, self).__init__(num_inputs)
        # according to linreg, grad is a vector with 3 dimension so
        self.S = np.zeros(num_inputs+1)
        
    def sgd_AdaGrad(self, grad, lr, sigma = 1E-6):
        # update adagrad vector
        self.S += grad ** 2
        # update parameters
        adagrad = (lr / np.square(self.S + sigma)) * grad
        self.w -= adagrad[:2]
        self.b -= adagrad[-1]




"""RMSProp- little improvement for adaGrad, avoid too small learning rate """
class RMSPropLinreg(linReg):
    def __init__(self, num_inputs):
        super(RMSPropLinreg, self).__init__(num_inputs)
        # according to linreg, grad is a vector with 3 dimension so
        self.S1 = np.zeros(num_inputs)
        self.S2 = np.zeros(1)
        self.gama = 0.9
        
        

    def sgd_RMSProp(self, grad, lr, sigma = 1E-6):
        self.S1 = self.gama*self.S1 + ((1-self.gama)*grad**2)[:2]
        self.S2 = self.gama*self.S2 + ((1-self.gama)*grad**2)[-1]
        
        # update parameters
        self.w -= (lr / np.square(self.S1 + sigma)) * grad[:2]
        self.b -= (lr / np.square(self.S2 + sigma)) * grad[-1]



"""AdaDelta Solving the problem when it's hard to find global optimization"""
class AdaDeltaLinreg(linReg):
    def __init__(self, num_inputs):
        super(AdaDeltaLinreg, self).__init__(num_inputs)
        self.S1 = np.zeros(2)
        self.S2 = np.zeros(1)
        self.delta = np.zeros(num_inputs+1)


    def sgd_AdaDelta(self, grad, sigma = 1E-5, ro=0.9):
        # update S
        self.S1 = ro*self.S1 + ((1-ro)*grad**2)[:2]
        self.S2 = ro*self.S2 + ((1-ro)*grad**2)[-1]

        #fix grad
        grad1 = np.square((self.delta[:2]+sigma)/(self.S1+sigma)) * grad[:2]
        grad2 = np.square((self.delta[-1]+sigma)/(self.S2+sigma)) * grad[-1]

        # update parameters
        self.w -= grad1
        self.b -= grad2

        # upadte delta
        self.delta = ro*self.delta + (1-ro)*np.concatenate([grad1, grad2])**2



"""Adam: RMSProp-Improvement for batch grad"""
class AdamLinreg(linReg):
    def __init__(self, num_inputs):
        super(AdamLinreg, self).__init__(num_inputs)
        self.S = np.zeros(num_inputs+1)
        self.V = np.zeros(num_inputs+1)
        

    def sgd_Adam(self, grad, lr, beta1=0.9, beta2=0.999, sigma=1E-6):
        self.V = beta1*self.V + (1-beta1)*grad
        self.S = beta2*self.S + (1-beta2) * grad**2

        ### bias fix
        """note: regarding the large timestep here, we do not implement bias fix
        which because small `1-beta2` will make self.S overflow
        """
        #self.V /= 1-beta1
        #self.S /= 1-beta2

        # fix grad
        grad = (lr*self.V)/(np.square(self.S)+sigma) * grad

        # update parameters
        self.w -= grad[:2]
        self.b -= grad[-1]