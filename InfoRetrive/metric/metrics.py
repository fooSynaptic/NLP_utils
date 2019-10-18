# encoding = utf-8
# /usr/bin/python3

import numpy as np


def map(preds, altTheme = 3):
    """Metric of map for search engine or recommondation"""
    assert len(preds) > 0 and len(preds[0]) >= altTheme, \
        "Un-resolvable parameters..."

    precisionS = np.zeros(len(preds), dtype = float)
    for i in range(len(preds)):
        acc = preds[i]
        precisionS[i] = sum(float(sum(acc[:j]))/float(j) for j in range(1, altTheme+1) if acc[j-1]==1)
    return np.mean(precisionS / altTheme)



def NDCG(idealabels, preds, k = 3):
    """Metric of NDCG for search engine or recommondation"""
    def _dcg_i(label, i):
        return 2**(label-1) / np.log2(i+1)

    ### dcg
    dcglist = np.zeros(len(preds), dtype=float)
    for i in range(len(preds)):
        pred = preds[i]
        dcglist[i] = sum(_dcg_i(pred[j], j+1) for j in range(k) if pred[j]==1)

    ### idcg
    idcglist = np.zeros(len(idealabels), dtype=float)
    for i in range(len(idealabels)):
        pred = idealabels[i]
        idcglist[i] = sum(_dcg_i(pred[j], j+1) for j in range(k) if pred[j]==1)

    # Ndcg
    return np.mean(dcglist / idcglist)







def testcase():
    arr1 = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1]
    ]

    arr2 = [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 0]
    ]


    res1 = map(arr1)
    print(res1)

    res2 = NDCG(arr2, arr1)
    print(res2)


#testcase()

        


