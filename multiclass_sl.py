from sklearn.svm import LinearSVC
#from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier

#from sklearn.feature_extraction.text import TfidfTransformer
#transformer = TfidfTransformer(smooth_idf=False)
#transformer.fit_transform(count)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
#vectorizer.fit_transform(corpus)



class Classifier():
    def __init__(self, xinput, yinput):
        assert isinstance(xinput, list)
        xinput = self.vec_encoder(xinput)
        self.label_idx = {}
        self.idx_label = {}
        
        for idx, label in enumerate(set(yinput)):
            self.label_idx[label] = idx
            self.idx_label[idx] = label
        
        yinput = [self.label_idx[label] for label in yinput]
        self.classifer = OneVsRestClassifier(LinearSVC(random_state=0)).\
            fit(xinput, yinput)

    def infer(self, text):
        vec = self.vec_encoder(text)
        res = self.classifer.predict(vec)
        return self.idx_label[res[0]]

    def vec_encoder(self, sentence):
        if isinstance(sentence, str):
            return vectorizer.transform([sentence]).toarray()
        elif isinstance(sentence, list):
            return vectorizer.fit_transform(sentence).toarray()


'''
corpus = ['This is the first document.', 'This is the second second document.', 'And the third one.', 'Is this the first document?']

labels = ['first', 'sec','third','first']

#fit model
model = Classifier(corpus, labels)
print(model.infer('i want the first document'))
'''
