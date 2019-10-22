# encoding = utf-8
# /usr/bin/python3

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
import jieba
import numpy as np
from collections import defaultdict
from pprint import pprint



# Use token freaquency (raw term count) features for MF.
print("Extracting tf features for MF...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features = 1000,
                                stop_words=None)



def features(corpus, tokenization = True, tokenizer = None):
    if tokenization and tokenizer is not None:
        for i in range(len(corpus)):
            corpus[i] = ' '.join(list(tokenizer(corpus[i])))

    tf = tf_vectorizer.fit_transform(corpus)
    return tf



def mf_cluster(docfeatures, n_components, cluster = None):
    if cluster is None:
        raise Exception("You should initilize cluster first...")

    mf = cluster(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(docfeatures)
    
    feature_names = tf_vectorizer.get_feature_names()
    resmf = mf.transform(docfeatures)


    topicMap = defaultdict(list)
    for i, x in enumerate(resmf):
        topic = np.argmax(x)
        topicMap[topic].append(i)

    return topicMap
    
    


def tescase():
    texts = [line['text'] for line in eval(\
    open('../../../nlp_project/text_struc/data/186_20190109_1400_channel_0.txt').read())]
    doc_features = features(texts[:], tokenizer=jieba.cut)
    
    clusterMap = mf_cluster(doc_features, 10, NMF)

    textCluster = defaultdict(list)

    for t in clusterMap.keys():
        for i in clusterMap[t]:
            textCluster[t].append(texts[i])


    for k in textCluster:
        pprint(textCluster[k])



if __name__ == '__main__':
    ### testcase
    #testcase()
    pass
    