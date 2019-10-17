# encoding = utf-8
# /usr/bin/python3
from utils import *
import os

os.sys.path.append('../crf_from_scratch/')
from models import linearChainCRF
import numpy as np
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import random


random.seed(1234)



def train(train_data, featrue2id, label2id, id2label, model_path, perpetualpath = None, epochs = 20):
    """Problem 2: given input and labels, estimate the parameters"""
    # perpetual for feature mapping
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    fl, flrev, ff = perpetualpath
    joblib.dump(label2id, fl)
    joblib.dump(featrue2id, ff)
    joblib.dump(id2label, flrev)

    ### intial random features
    crf.W[:] = np.random.uniform(-1, 1, size=crf.W.shape)

    ### learning the parameters
    crf.sgd(train_data, iters = epochs, saveParam = True, modelpath = model_path)

    print("Done for parameter trainning...")



def evaluate(eval_data, fearure2idPath, label2idPath, id2labelPath, model_path, verbose = False):
    """eval model"""
    crf.W[:] = joblib.load(os.path.join(model_path, 'crf_W.pkl'))
    label2id, attr2id, id2label = joblib.load(label2idPath), \
                                joblib.load(fearure2idPath), \
                                joblib.load(id2labelPath)


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

    print("Done with evaluation...")




def run():
    data = load_data('dataset/train_data.txt', proportion=0.1)[:50]

    #p = int(len(data) * 0.9)
    p = 45
    train_data, eval_data = data[:p], data[p:]
    print("train sample num: {}, test samples num: {}".format(len(train_data), len(eval_data)))

    ### feature mapping initial
    label2id, attr2id = updateFieldNode(data)
    id2label = {}
    for k, v in label2id.items(): id2label[v] = k
    ### initialize crf
    model_path = 'linearCRF'


    global crf
    crf = linearChainCRF(attr2id, label2id)

    labelmapPath = os.path.join(model_path, 'labelmap.pkl')
    featuremapPath = os.path.join(model_path, 'featuremap.pkl')
    revlabelmapPath = os.path.join(model_path, 'rev_labelmap.pkl')

    train(train_data, attr2id, label2id, id2label, model_path, \
        [labelmapPath, revlabelmapPath, featuremapPath], epochs= 3)


    evaluate(eval_data, featuremapPath, labelmapPath, revlabelmapPath, model_path)



if __name__ == "__main__":
    run()