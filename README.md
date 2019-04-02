# Py_utils
Wrappers for NLP programming.


# add edit distance for spell checking

**test case:**
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


**result:**
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
