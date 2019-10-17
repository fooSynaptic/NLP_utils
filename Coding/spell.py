import re
from collections import Counter
from jieba import cut


class speller():
    def __init__(self, corpus_path):
        words, self.CHARs = self.words(open(corpus_path).read())
        self.WORDS = Counter(dict(zip(words, self.CHARs)))

    def words(self, text):
        text = 
        return [cut(re.sub("[a-z0-9_ ]", "", text.lower()))],\
            re.sub("[a-z0-9_ ]", "", text.lower()).split()

    def P(self, word): 
        "Probability of `word`."
        N=sum(self.WORDS.values())
        return self.WORDS[word] / N

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word): 
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) \
            or self.known(self.edits2(word)) or [word])

    def known(self, words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = self.CHARs
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))


Optimizer = speller('./data/content-title.txt')
Optimizer.correction('郭家领导人')
