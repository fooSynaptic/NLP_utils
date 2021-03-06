# encoding=utf-8
# /usr/bin/python3
from bs4 import BeautifulSoup as bs
import bs4
import re
import numpy as np



"""Implementation tools... """




def tokenfeatures(word):
    """ Obsevation state features """
    features = []

    ###
    features.append("token==" + word)

    ###
    mid = len(word)//2
    if mid > 0:
        features.append('Swith+'+word[:mid])
        features.append('Ewith+'+word[mid:])
    
    ### 4
    fsymolic = re.findall('[a-zA-Z0-9_]', word)
    if len(fsymolic) > 0:
        features.extend(["contains" + c for c in fsymolic])
    return features



def Transferfeature(seq):
    """ Transfer features """
    seq.insert(0, ['seq_start'])
    seq.append(['seq_end'])
    for i in range(len(seq)):
        seq[i].extend(tokenfeatures(seq[i][0]))
    if True:
        # previous token feature
        for i in range(1, len(seq)):
            seq[i].extend('@-1:'+f for f in seq[i-1][1:] \
                if not (f.startswith('@+1:') or f.startswith('@-1:')))
        # next token feature
        for i in range(len(seq)-1):
            seq[i].extend('@+1:'+f for f in seq[i+1][1:] \
                if not (f.startswith('@+1:') or f.startswith('@-1:')))
    return seq



def parser(line):
    """parse a tagged sentence pair."""
    tgt, src = line.split('\t')

    src = [[token] for token in src.split()]
    tgt = tgt.split()

    tgt.insert(0, 'seqStart')
    tgt.append('seqEnd')

    ###
    return src, tgt




def load_data(path, proportion=0.1):
    """load src data given proportion"""
    data = []
    for line in open(path).readlines():
        line = line.strip()

        x, y = parser(line)
        pair = (Transferfeature(x), y)
        data.append(pair)

    return data[:int(proportion * len(data))]



def updateFieldNode(data):
    """Loading feature map configure"""
    labels, attributes = set(), set()
    for ftoken, label in data:
        labels.update(label)
        attributes.update(word for line in ftoken for word in line)

    #attributes.update(['seqStart', 'seqEnd'])
    label2id, attr2id = dict(), dict()
    for _id, label in enumerate(labels):
        label2id[label] = _id

    for _id, attr in enumerate(attributes):
        attr2id[attr] = _id
    return label2id, attr2id


def logsumexp(a):
    """
    Compute the log of the sum of exponentials of an array ``a``, :math:`\log(\exp(a_0) + \exp(a_1) + ...)`
    """
    ### introduce max(a) to avoid overflow.
    A = a.max()
    return A + np.log((np.exp(a-A)).sum())


def sample(x):
    cdf = x.cumsum()
    Z = cdf[-1]
    u = uniform()
    return cdf.searchsorted(u * Z)



class FeatureTable():
    def __init__(self, tokenlist, xnode, ynode):
        self.seq = []
        for tokenfeatures in tokenlist:
            # tokenfeatures[0] is the raw token, others are token attributes.
            self.seq.append(np.array([xnode[f] for f in tokenfeatures], dtype=np.int32))

        self.xdim, self.ydim = len(xnode), len(ynode)

    def __getitem__(self, item):
        t, yp, y = item
        ftoken = self.seq[t]
        if yp is not None: 
            ftoken = np.append(ftoken, yp)

        return [(f, y) for f in ftoken]
        
        




'''
class FeatureTable():
    def __init__(self, tokenlist, xnode, ynode):
        self.seq = []
        for tokenfeatures in tokenlist:
            # tokenfeatures[0] is the raw token, others are token attributes.
            self.seq.append(np.array([xnode[f] for f in tokenfeatures], dtype=np.int32))

        self.xdim, self.ydim = len(xnode), len(ynode)

    def __getitem__(self, item):
        t, yp, y = item
        ftoken = self.seq[t]
        if yp is not None:
            return np.append(ftoken, yp) + y*self.xdim
        else:
            return ftoken + y*self.xdim
'''

