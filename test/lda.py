# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
#from sklearn.datasets import fetch_20newsgroups

n_samples = 2000
n_features = 1000
n_components = 77
n_top_words = 10

stopwords = [x.strip() for x in open('/Users/ajmd/Desktop/stopwords.txt')\
      .readlines()]

stopwords += '你好 您好 一个 两个 今天 明天 上次 昨天 现在 一次 不是 \
      大家 当时 后面 时间 这回 一下 一些 一回事 老师 电话 打电话 热线\
             谢谢 先生 好好 一起 一直 没有'.split()
#print(len(stopwords), stopwords[:100])

def print_top_words(model, feature_names, n_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
        topic_dict[topic_idx] = [feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]]
    #print([topic_dict[x] for x in topic_dict])
    return topic_dict


def infer(res, lda_topics):
      vec = []
      for i, x in enumerate(res):
            tokens = set(data_samples[i].split())
            topic_tokens = lda_topics[np.argmax(x)]
            tokens = [x for x in tokens if x in topic_tokens]
            map_rate = round(len(tokens)/len(topic_tokens), 3)
            vec.append(map_rate)
            print(i, 'Topic-',np.argmax(x), 'maprate-', map_rate, tokens)
      print('average map-rate:', sum(vec)/len(vec))
      return sum(vec)/len(vec)



# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")

import os

data_samples = [x.strip() for x in open('./summer_result.txt').readlines()]

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words=stopwords)
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)


# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words=stopwords)
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print()

# Fit the NMF model
print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)


print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#print_top_words(nmf, tfidf_feature_names, n_top_words)

resultnmf = nmf.transform(tfidf)
nmf_topic = print_top_words(nmf, tfidf_feature_names, n_top_words)
print(resultnmf, nmf_topic)
infer(resultnmf, nmf_topic)

# Fit the NMF model

nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#print_top_words(nmf, tfidf_feature_names, n_top_words)

resultnmf = nmf.transform(tfidf)
nmf_topic = print_top_words(nmf, tfidf_feature_names, n_top_words)
infer(resultnmf, nmf_topic)

'''
stores = []
for i in range(1, 300):
      n_components = i
      nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)

      resultnmf = nmf.transform(tfidf)
      nmf_topic = print_top_words(nmf, tfidf_feature_names, n_top_words)
      stores.append([i, infer(resultnmf, nmf_topic)])
print(sorted(stores, key=lambda stores:stores[1]))
'''



print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
lda_topics = print_top_words(lda, tf_feature_names, n_top_words)

res = lda.transform(tfidf)


infer(res, lda_topics)


stores = []
for i in [65, 77, 117]:
      n_components = i
      lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

      lda.fit(tfidf)
      print("done in %0.3fs." % (time() - t0))

      print("\nTopics in LDA model:")
      tf_feature_names = tf_vectorizer.get_feature_names()
      lda_topics = print_top_words(lda, tf_feature_names, n_top_words)
      import numpy as np
      res = lda.transform(tfidf)


      #infer(res, lda_topics)
      stores.append([i, infer(res, lda_topics), lda.perplexity(tfidf)])
print(stores)
print(sorted(stores, key=lambda stores:stores[1]))
