# encoding = utf-8
# /usr/bin/python3

from utils import *
import sys
from models import linearChainCRF
import numpy as np
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import random

random.seed(1234)



def run():
    dataPath = './data/tagged_references.txt'
    data = load_data(dataPath, proportion=0.1)
    p = int(len(data)*0.9)
    train_data, eval_data = data[:p], data[p:]
    print("train sample num: {}, test samples num: {}".format(len(train_data), len(eval_data)))

    ### feature mapping initial
    label2id, attr2id = updateFieldNode(data)
    id2label = {}
    for k, v in label2id.items(): id2label[v] = k
    ### initialize crf
    crf = linearChainCRF(attr2id, label2id)

    train = True
    if train:
        joblib.dump(label2id, './linearCRF/labelmap.pkl')
        joblib.dump(attr2id, './linearCRF/featuremap.pkl')
        joblib.dump(id2label, './linearCRF/rev_labelmap.pkl')

        """Problem 2: given input and labels, estimate the parameters"""
        ### intial random features
        crf.W[:] = np.random.uniform(-1,1,size=crf.W.shape)

        ### learning the parameters
        crf.sgd(train_data, iters=10, saveParam = True)
    else:
        crf.W[:] = joblib.load('linearCRF/crf_w.pkl')
        label2id, attr2id, id2label = joblib.load('./linearCRF/labelmap.pkl'), \
            joblib.load('./linearCRF/featuremap.pkl'), \
                joblib.load('./linearCRF/rev_labelmap.pkl')


    ### logit estimate and label inference
    EVAL = True; verbose = True
    if EVAL:
        llogit, f1 = 0., 0.
        n = len(eval_data)
        for x, labels in eval_data:
            ftoken = FeatureTable(x, attr2id, label2id)
            flabel = crf.edgefeatures(ftoken, [label2id[label] for label in labels])
            """Problem1: estimate the probability given model parameters and target label"""
            Prob_hat = crf.logitEstimator(ftoken, flabel)

            """Probkem3: inference the target label given model parameters abd input"""
            labelpredict = [id2label[t] for t in crf.vertebi_inference(ftoken)][1:-1]
            raw_label = labels[1:-1]
            f1 += classification_report(raw_label, labelpredict, output_dict=True)['weighted avg']['f1-score']
            if verbose:
                print("src: ", ' '.join(raw_label))
                print('tgt: ', ' '.join(labelpredict))
            llogit += Prob_hat
        llogit /= n
        f1 /= n

        print("Average estimator logit likelyhood is {}.".format(llogit))
        print("Average f1 is {}.".format(f1))
 



if __name__ == '__main__':
    run()