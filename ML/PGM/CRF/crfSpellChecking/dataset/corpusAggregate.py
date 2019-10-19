# encoding=utf-8
# /usr/bin/python3

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups


n_features = 10
# Use tf (token freaquency ) features for LDA.
print("Extracting tf features for text Vectoring...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')



def textfeature():
