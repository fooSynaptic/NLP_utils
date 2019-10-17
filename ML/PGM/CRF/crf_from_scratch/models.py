# encoding=utf-8
# /usr/bin/python3

"""crf implementation from scratch"""

import numpy as np
from collections import defaultdict
from numpy import empty, zeros, ones, log, exp, add, int32
from numpy.random import uniform
from utils import *
import os
import random
random.seed(1234)

from sklearn.externals import joblib
from tqdm import tqdm





class linearChainCRF():
    def __init__(self, xnode, ynode):
        """
        xnode: the mapping from observation features to int
        ynode: the mapping from target label to int
        """
        self.K = len(ynode)              # size of target label
        self.M = len(xnode)	        # total feature dimension length
        self.featureNodes, self.labelNodes = xnode, ynode
        self.W = np.zeros((self.M, self.K))
        print("feature dimension: ", self.W.shape)

    
    def currentField(self, x):
        """calculate potential parameter given current obeservation data. """
        """x is a taggedFeatureTable object"""
        N, K, W = x.seq.__len__(), self.K, self.W

        # estimate
        g0 = empty(K)
        g = empty((N-1, K, K))    #at timestep n, the logit of i-th label transfer to j-th label

        for y in range(K):
            # at step 0, the logit of label k (from time -1)
            g0[y] = sum(W[idx[0], idx[1]] for idx in x[0, None, y])
            #g0[y] = W[x[0, None, y]].sum()

        for t in range(1, N):
            for y in range(K):
                for yp in range(K):
                    g[t-1,yp,y] = sum(W[idx[0], idx[1]] for idx in x[t, yp, y])

        return g0, g


    
    def vertebi_inference(self, x):
        """sequence labeling, inference the most likely label given x"""
        """x is a taggedFeatureTable object"""
        N, K = x.seq.__len__(), self.K
        g0, g = self.currentField(x)
        B = ones((N, K), dtype=int32) * -1

        # compute max-marginals and backtrace matrix
        V = g0
        for t in range(1, N):
            U = empty(K)            # at time step t, only record the maxlogit pre label
            for y in range(K):
                w = V + g[t-1,:,y]  # at time t, the logit of all labels in t-1 transfer to label y
                B[t,y] = b = w.argmax()     # recore which label in time t-1 have the max logit
                U[y] = w[b]
            V = U

        # extract the best path by brack-tracking
        y = V.argmax()
        trace = []
        for t in range(N-1, -1, -1):
            trace.append(y)
            y = B[t, y]

        return trace[::-1]


    def logitEstimator(self, x, targetLabel):
        """estimate label likely-hood given W"""
        N = x.seq.__len__()
        K = self.K
        W = self.W
        g0, g = self.currentField(x)
        a = self.forward(g0, g, N, K)
        logZ = logsumexp(a[N-1, :])
        #return sum(W[k] for k in targetLabel) - logZ
        return sum(W[idx[0], idx[1]] for idx in targetLabel) - logZ


    def forward(self, g0, g, N, K):
        """forward stepwise"""
        a = zeros((N, K))
        a[0, :] = g0

        for t in range(1, N):
            yp = a[t-1, :]
            for y in range(K):
                a[t, y] = logsumexp(yp + g[t-1, :, y])
        
        return a


    def backward(self, g, N, K):
        """backword stepwise"""
        b = zeros((N, K))

        for t in range(N-2, -1, -1):
            yp = b[t+1, :]
            for y in range(K):
                b[t, y] = logsumexp(yp + g[t, y, :])

        return b


    def Exp(self, x):
        """expectation of x"""
        N = x.seq.__len__()
        K = self.K

        g0, g = self.currentField(x)

        a = self.forward(g0, g, N, K)
        b = self.backward(g, N, K)
        ### scale Z
        logZ = logsumexp(a[N-1, :])

        ans = dict()
        # here e = exp(a + b - logZ)
        # at step 0, a == g0 
        e0 = exp(g0 + b[0,:] - logZ).clip(0., 1.)
        for y in range(K):
            prob = float(e0[y])
            for k in x[0, None, y]:
                # the k-pos logit from step -1 transfer to label y
                ans[k] = ans.get(k, 0.) + prob


        for t in range(1, N):
            """
            Outer: broadcast: not element-wise operation
            """
            # e_i = {(forward[t-1] + backward[t])*P(y|x)_t-1}/ logZ	    logit of time i
            # or exp(a[t-1,yp] + g[t-1,yp,y] + b[t,y] - logZ)
            ei = exp((add.outer(a[t-1,:], b[t,:]) + g[t-1,:,:] - logZ)).clip(0., 1.)
            #ei = exp(a[t-1,:] + g[t-1,:,:] + b[t,:] - logZ).clip(0., 1.)
            #print(ei.shape)
            for yp in range(K):
                for y in range(K):
                   prob = float(ei[yp, y])
                   for k in x[t, yp, y]:
                       ans[tuple(k)] = ans.get(k, 0.) + prob

        return ans


    def edgefeatures(self, x, y):
        """
        This is important
        for our raw input t1, t2, t3, t4 ...
        transfer the token sequence information into {f1, ... ft}1, {f1, ... ft}2
        """
        assert len(x.seq) == len(y), "seq to label not alignmentted..."
        features = list(x[0, None, y[0]])   #tuple list
        
        for t in range(1, x.seq.__len__()):
            for idx in x[t, y[t-1], y[t]]:
                features.append(idx)

        return features


    def sgd(self, data, iters = 20, Nwarm_up = 10, saveParam = False, modelpath = 'linearCRF'):
        W = self.W  ## reference
        for i in range(iters):
            print("Epochs: ", i)
            lr_rate = Nwarm_up/(i+1)**0.501
            for x, labels in tqdm(data):
                ftoken = FeatureTable(x, self.featureNodes, self.labelNodes)
                flabel = self.edgefeatures(ftoken, [self.labelNodes[label] for label in labels])
                for k, explogit in self.Exp(ftoken).items():
                    W[k[0], k[1]] -= lr_rate * explogit
                for k in flabel:
                    W[k[0], k[1]] += lr_rate

        if saveParam:
            if not os.path.exists(modelpath):
                os.mkdir(modelpath)
            joblib.dump(self.W, os.path.join(modelpath, 'crf_W.pkl'))

