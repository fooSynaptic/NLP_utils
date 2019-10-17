# encoding = utf-8
# /usr/bin/python3
from jieba import cut
import numpy as np
from sklearn.externals import joblib
from random import sample, randint



def preprocess(datapath):
    vocabs = set()
    charVocabs = set()
    corpus = []
    lengths = []
    for line in open(datapath).readlines():
        line = line.strip().split()[1].split('。')[0] + '。'
        lengths.append(len(line))
        segline = cut(line)

        charVocabs.update(list(line))
        vocabs.update(segline)
        corpus.append(' '.join(cut(line)))


    with open('./rawdata.txt', 'w') as f:
        threshould = np.median(lengths)
        for l in corpus:
            if len(list(l.split())) > threshould:
                continue

            f.write(l + '\n')

    print("average length of sentences: ", np.mean(lengths), np.median(lengths))

    print("Save vocabulary... length: ", len(vocabs))

    joblib.dump(vocabs, 'vocabs.pkl')
    joblib.dump(charVocabs, 'charVocabs.pkl')


    print("Done...")










def generateTraindata(spellerrorFlod = 3, Spelling = 'word'):
    src_data = [line.strip() for line in open('rawdata.txt').readlines()]
    tgt_data = []

    charVocabs = joblib.load('charVocabs.pkl')
    vocabs = joblib.load('vocabs.pkl')
    n1, n2 = len(charVocabs), len(vocabs)


    # build token mapping
    id2char, id2token = {}, {}
    for i, char in enumerate(charVocabs):
        id2char[i] = char

    for i, token in enumerate(vocabs):
        id2token[i] = token



    for sent in src_data:
        print('src: ', sent)
        features = sent.split()
        k = len(features)
        change = set(sample([i for i in range(k)], k//spellerrorFlod))
        features_generated = []

        for i in range(k):
            if i in change:
                change_char_idx = randint(0, len(features[i])-1)
                tgt_char_idx = randint(0, n1-1)
                tgt_word_idx = randint(0, n2-1)
                if Spelling == 'word':
                    features_generated.append(id2token[tgt_word_idx])
                else:
                    #features[i][change_char_idx] = id2char[tgt_char_idx]
                    features[i] = features[i].replace(features[i][change_char_idx], id2char[tgt_char_idx])
                    features_generated.append(features[i])
            else:
                features_generated.append(features[i])

        newSent = ' '.join(features_generated)
        tgt_data.append(newSent)

    assert len(src_data) == len(tgt_data)

    with open("./train_data.txt", 'w') as f:
        for i in range(len(src_data)):
            f.write(src_data[i] + '\t' + tgt_data[i] + '\n')

    print("Generate {} Spelling error data accomplished...".format(Spelling))
            

#preprocess('cnews.train.txt')
generateTraindata()