import logging
import numpy as np
from collections import Counter
import math


logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        #filename='/home/URL/client/test_log.log',
                        filemode='a')



MEDICAL_DB = '/Users/ajmd/code/gitrep/funNLP/data/医学词库/THUOCL_medical.txt'


def bleu(pred_tokens, label_tokens):
    if '*' not in pred_tokens: k = len(pred_tokens) - 1
    else: k = len(pred_tokens.replace('*', ''))
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches = 0
        for i in range(len_pred - n + 1):
            if pred_tokens[i: i + n] in label_tokens:
                num_matches += 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score



def EditDis(src, tgt):
    len1, len2 = len(src), len(tgt)
    #print(len1, len2)
    dp = np.zeros([len1, len2])
    #intializing
    for i in range(0, len1):
        for j in range(0, len2):
            dp[i][j] = float('inf')
    
    for i in range(0, len1):
        dp[i, 0] = i
    
    for j in range(0, len2):
        dp[0, j] = j

    for i in range(0, len1):
        for j in range(1, len2):
            if src[i] == tgt[j]:
                flag = 0
            else:
                flag = 1

            dp[i, j]=min(dp[i-1, j]+1,min(dp[i,j-1]+1,dp[i-1,j-1]+flag))
    return dp[len1-1, len2-1]


#print(EditDis('胃食管反留' ,'胃食管反流'))
#print(EditDis('胃食管反留' ,'严重性胃病'))


class corrector():
    def __init__(self, dbpath):
        tokens = [x.strip().split() for x in open(dbpath).readlines()]
        words, freqs = [x[0] for x in tokens], [int(x[1]) for x in tokens]
        self.WORDS = Counter(dict(zip(words, freqs)))

    def candidate(self, word):
        if len(word)>2: mapval = len(word)-2
        chars = set(word)
        candidates = {}
        for token in self.WORDS.keys():
            cnt = len(set([x for x in token if x in chars]))
            if cnt >= 2 and len(token) >= len(word):
                #candidates[token] = cnt
                try:       
                    candidates[token] = bleu(word, token)
                except:
                    logging.debug(token +'\t' + word)
            else: pass
            
        return candidates
    
    def correct(self, word):
        candidates = self.candidate(word)
        #print('candidates:', candidates)
        min_dis, tgt_tokens = len(word), []
        for tgt, _ in candidates.items():
            dis = EditDis(tgt, word)
            if dis <= min_dis:
                #print(tgt, candidates[tgt], dis*0.1)
                candidates[tgt] += dis*0.01
                tgt_tokens.append(tgt)
            else: pass
        #for token in tgt_tokens: print(token, candidates.get(token))
        return max(tgt_tokens, key = candidates.get) if tgt_tokens else None

def testcase():       
    Optimizer = corrector(MEDICAL_DB)
    candidates = ['性疾病', '血管疾病', '性肝病']
    for word in candidates:
        logging.info(Optimizer.correct(word))
'''


logging.info(Optimizer.distance('胃食管反*'))
logging.info(Optimizer.distance('*子鉴定'))
logging.info(Optimizer.distance('子鉴定'))
logging.info(Optimizer.distance('囊恶性肿瘤'))
logging.info(Optimizer.distance('他定类**'))
logging.info(Optimizer.distance('*他定'))
logging.info(Optimizer.distance('**他定'))
'''


'''
int EditDis()
{
    int len1 = strlen(a+1);
    int len2 = strlen(b+1);
    //初始化
    for(int i=1;i<=len1;i++)
        for(int j=1;j<=len2;j++)
            dp[i][j] = INF;
    for(int i=1;i<=len1;i++)
        dp[i][0] = i;
    for(int j=1;j<=len2;j++)
        dp[0][j] = j;
    for(int i=1;i<=len1;i++)
    {
        for(int j=1;j<=len2;j++)
        {
            int flag;
            if(a[i]==b[j])
                flag=0;
            else
                flag=1;
            dp[i][j]=min(dp[i-1][j]+1,min(dp[i][j-1]+1,dp[i-1][j-1]+flag));
            //dp[i-1][j]+1表示删掉字符串a最后一个字符a[i]
            //dp[i][j-1]+1表示给字符串添加b最后一个字符
            //dp[i-1][j-1]+flag表示改变,相同则不需操作次数,不同则需要,用flag记录
        }
    }
    return dp[len1][len2];
}
'''
