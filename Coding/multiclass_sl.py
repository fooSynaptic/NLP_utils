from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


from sklearn.preprocessing import LabelEncoder
labelInfer = LabelEncoder()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()



class Classifier():
    def __init__(self, xinput, yinput):
        assert isinstance(xinput, list)
        xinput = self.vec_encoder(xinput)
        self.label_idx = {}
        self.idx_label = {}
        
        labelInfer.fit(yinput)
        for idx, label in enumerate(set(yinput)):
            self.label_idx[label] = idx
            self.idx_label[idx] = label
        yinput = labelInfer.transform(yinput)
        #yinput = [self.label_idx[label] for label in yinput]

        self.classifer = OneVsRestClassifier(LinearSVC(random_state=0)).\
            fit(xinput, yinput)

    def infer(self, text):
        vec = self.vec_encoder(text)
        res = self.classifer.predict(vec)
        return labelInfer.inverse_transform([res[0]])

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
