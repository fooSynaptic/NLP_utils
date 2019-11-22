# NLP_utils
NLP programming and machine learning implementations.

***Machine learning from scratch***
---
- [Linear regression with parameter learn by Gradient descent.](https://github.com/fooSynaptic/Py_utils/blob/master/ML/regression/linearRegression.py)
- [Conditional Random Field(CRF) probability solving Implementation.](https://github.com/fooSynaptic/NLP_utils/blob/master/ML/PGM/CRF/crfProb_logit.py)
- [Various Optimization algo Implemented in LR(from sgd to adam).](https://github.com/fooSynaptic/NLP_utils/blob/master/ML/regression/optimizer.py)
- [Crf implement from scratch(f1 = 0.916 over reference tagging dataset from https://people.cs.umass.edu/~mccallum/data.html).](https://github.com/fooSynaptic/NLP_utils/tree/master/ML/PGM/CRF/crf_from_scratch)
- [Chinese sentence sepll checking with CRF labeling.](https://github.com/fooSynaptic/NLP_utils/tree/master/ML/PGM/CRF/crfSpellChecking)
- [IBM translation model-1](https://github.com/fooSynaptic/NLP_utils/blob/master/ML/machineTranslation/SMT/IBM_translationModel.py)





# Update edit distance spell checking for Medical words
- run `python edit_distance.py`
- run `python spell.py`

**Test case:**
```
def testcase():          
    candidates = ['性疾病', '血管疾病', '性肝病']
    for word in candidates:
        logging.info(Optimizer.correct(word))

testcase()

logging.info(Optimizer.correct('胃食管反*'))
logging.info(Optimizer.correct('*子鉴定'))
logging.info(Optimizer.correct('子鉴定'))
logging.info(Optimizer.correct('囊恶性肿瘤'))
logging.info(Optimizer.correct('他定类**'))
logging.info(Optimizer.correct('*他定'))
logging.info(Optimizer.correct('**他定'))
```
**Result:**
```
Tue, 02 Apr 2019 17:26:03 edit_distance.py[line:103] INFO 药源性疾病
Tue, 02 Apr 2019 17:26:03 edit_distance.py[line:103] INFO 心脑血管疾病
Tue, 02 Apr 2019 17:26:03 edit_distance.py[line:103] INFO 慢性肝病
Tue, 02 Apr 2019 17:26:03 edit_distance.py[line:107] INFO 胃食管反流
Tue, 02 Apr 2019 17:26:03 edit_distance.py[line:108] INFO 亲子鉴定
Tue, 02 Apr 2019 17:26:03 edit_distance.py[line:109] INFO 亲子鉴定
Tue, 02 Apr 2019 17:26:03 edit_distance.py[line:110] INFO 精囊恶性肿瘤
Tue, 02 Apr 2019 17:26:03 edit_distance.py[line:111] INFO 奥洛他定滴眼液
Tue, 02 Apr 2019 17:26:04 edit_distance.py[line:112] INFO 头孢他定
Tue, 02 Apr 2019 17:26:04 edit_distance.py[line:113] INFO 头孢他定
```

# Update classfication based on Naive bayes.
- run `python bayesian.py`
**Result**

```
If we start from state of 1
The condition when 1 -> 2 -> S happend with prob of 0.04
If we start from state of -1
The condition when -1 -> 2 -> S happend with prob of 0.06
The most likely feature of fearure_3 with f1 and f2 is -1
```

# Update KD-tree for cluster algorithm
- usage: `python ./ML/KD_tree.py`

demo:
```
The levels denote the tree depth, same level means they stay in parallel,'
'    and the next level denote the parent and child information, in our code,'
     for two stacked node, right node first and left node second...
 level0  level1  level2  level3  level4  level5
 False
         [[5 4]]
                 [[2 3]](right node of (5, 4))
                 [[4 7]](left node of (5, 4))
         [[7 2]]
                 [[8 1]]
                 [[9 6]]
 ```
 - You can try it with differnt test_data as well as more dimension
