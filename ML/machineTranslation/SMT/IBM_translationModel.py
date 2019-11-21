# encoding = utf-8
# /usr/bin/python3

"""
## IBM translation model
整个目标是通过EM算法学习IBM model中的transition prob参数，
通过EM算法，可以在迭代统计平行语料的过程中，不断计算概率 t(e|f) 的似然概率，
并更新到参数中，以达到学习整个参数中最大alignment对应最大概率的目的.
"""

import numpy as np
from time import time 
from tqdm import tqdm
from sklearn.externals import joblib
import jieba
import re
import math
import collections



def IBM_TransModel(transPairs, srcvocabs, tgtvocabs, srcMapper, tgtMapper):
    """
    Input: set of sentence pairs (e, f)
    Output: translation prob. t(e|f)
    """
    beginTime = time()
    src_vocabSize, tgt_vocabSize = len(srcvocabs), len(tgtvocabs)

    ## initialize t(e|f) uniformly
    transProb = np.random.rand(src_vocabSize, tgt_vocabSize)

    for epoch in range(20):
        print("epochs: ", epoch)
        Cnt = np.zeros([src_vocabSize, tgt_vocabSize])
        # the expectation of existance of words
        Exp_src = np.zeros(src_vocabSize)
        Exp_src += 1E-20
        Exp_tgt = np.zeros(tgt_vocabSize)
        Exp_tgt += 1E-20

        # estimation
        for source, target in tqdm(zip(transPairs[0], transPairs[1])):
            ## compute normalization
            for tgtWord in target:
                Exp_tgt[tgtWord] = 0
                for srcWord in source:
                    Exp_tgt[tgtWord] += transProb[srcWord, tgtWord]


            ## collect counts
            for tgtWord in target:
                for srcWord in source:
                    Cnt[srcWord, tgtWord] += transProb[srcWord, tgtWord] \
                        / Exp_tgt[tgtWord]
                    Exp_src[srcWord] += transProb[srcWord, tgtWord] \
                        / Exp_tgt[tgtWord]

        # expectation
        ## estimate Probabilities
        for w1 in srcvocabs:
            for w2 in tgtvocabs:
                a, b = srcMapper[w1], tgtMapper[w2]
                transProb[a, b] = Cnt[a, b] / (Exp_src[a] + 1E-6)
                assert transProb[a, b] is not float('nan')

    endTime = time()
    print("time eclipse: ", endTime - beginTime)

    return transProb


def tokenization(line, mode = None):
    line = line.strip()
    if mode is None:
        return list(line)
    elif mode == "zh":
        return list(jieba.cut(line))
    elif mode == 'en':
        line = line.lower()
        tokens = list(line)
        for i in range(len(tokens)):
            if len(re.findall('[^a-z0-9]', tokens[i])) > 0:
                tokens[i] = ' '+tokens[i]
        line = ''.join(tokens)
        return line.split()


def translate(model, line, srcmapper, tgtmapper, unknowToken = "."):
    seq = [(srcmapper[x] if x in srcmapper else srcmapper[unknowToken]) for x in line]

    result = []
    for word in seq:
        srctoken = word
        ## greedy
        tgttoken = tgtmapper[np.argmax(model[srctoken, :])]
        result.append(tgttoken)

    return ' '.join(result)


def bleu(pred_tokens, label_tokens, k):
    """craft bleu realization"""
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def train():
    src_file = "./data/en-zh/train.tags.en-zh.en"
    tgt_file = "./data/data/NMT/en-zh/train.tags.en-zh.zh"

    rawSrc = [tokenization(line, mode = 'en') for line in open(src_file).readlines() if not line.startswith('<')]
    rawTgt = [tokenization(line, mode = 'zh') for line in open(tgt_file).readlines() if not line.startswith('<')]
    S, s_test = rawSrc[:10000], rawSrc[10000:11000]
    T, t_test = rawTgt[:10000], rawTgt[10000:11000]



    assert len(S) == len(T), "corpus not alignmented..."
    srcvocabs, tgtvocabs = set(), set()

    for sline in S: srcvocabs.update(sline)
    for tline in T: tgtvocabs.update(tline)

    src2id, id2src = dict(), dict()
    tgt2id, id2tgt = dict(), dict()

    for i, word in enumerate(srcvocabs):
        src2id[word] = i
        id2src[i] = word

    for i, word in enumerate(tgtvocabs):
        tgt2id[word] = i
        id2tgt[i] = word

    Strain = [[src2id[word] for word in line] for line in S]
    Ttrain = [[tgt2id[word] for word in line] for line in T]


    if not os.path.exists("ibmModel.pkl"):
        ### train
        model = IBM_TransModel([Strain, Ttrain], srcvocabs, tgtvocabs, src2id, tgt2id)
        joblib.dump(model, "ibmModel.pkl")
    else:
        model = joblib.load("ibmModel.pkl")

    ### translate(inference)
    resM = np.zeros([1000, 4])
    for i in range(1000):
        src, pred = s_test[i], translate(model, s_test[i], src2id, id2tgt)
        tgt = t_test[i]
        b1, b2, b3, b4 = bleu(pred, tgt, 1), bleu(pred, tgt, 2), bleu(pred, tgt, 3), bleu(pred, tgt, 4)
        #print("- src: ", S[i], '\n', "- tgt: ", translate(model, S[i], src2id, id2tgt))
        resM[i][0] = b1
        resM[i][1] = b2
        resM[i][2] = b3
        resM[i][3] = b4

    print("eval result:", np.mean(resM, axis=0))
    #joblib.dump(model, "ibmModel.pkl")




if __name__ == "__main__":
    train()













