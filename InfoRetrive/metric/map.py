# encoding = utf-8
# /usr/bin/python3

import numpy as np


def map(preds, altTheme = 3):
    """Metric of map for search engine or recommondation"""
    assert len(preds) > 0 and len(preds[0]) >= altTheme, \
        "Un-resolvable parameters..."

    precisionS = []
    for i in range(len(preds)):
        acc = preds[i]
        p = 0.
        for i in range(1, altTheme):
            p += sum(acc[:i]) / i
        p /= altTheme
    
        precisionS.append(p)

    return np.mean(precisionS)



def NDCG(idealabels, preds, k = 3):
    """Metric of NDCG for search engine or recommondation"""
    def _dcg_i(label, i):
        return 2**(label-1) / np.log2(i+1)


    dcglist = []
    for i in range(len(preds)):
        pred = preds[i]
        p = 0.
        for j in range(k):
            p += _dcg_i(pred[j], j+1)
        
        dcglist.append(p)

    idcglist = []
    for i in range(len(idealabels)):
        pred = idealabels[i]
        p = 0.
        for j in range(k):
            p += _dcg_i(pred[j], j+1)

        idcglist.append(p)

    return np.mean(np.array(dcglist) / np.array(idcglist))







def testcase():
    res1 = map([[1,1,0], [0, 0, 1], [0, 1, 1], [1, 1, 1]])
    print(res1)

    res2 = NDCG([[1,1,0], [0, 0, 1], [0, 1, 1], [1, 1, 1]], [[0,1,0], [0, 0, 0], [0, 1, 1], [1, 0, 1]])
    print(res2)


testcase()

        


