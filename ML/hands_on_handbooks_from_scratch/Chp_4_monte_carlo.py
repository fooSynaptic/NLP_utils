'''
Implementation for the Monte_carlo algorithms in chaper 4 of AI handbook
- ref: http://www.huaxiaozhuan.com/%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80/chapters/4_monte_carlo.html
'''

from math import pi
import numpy as np
import random
from random import randint
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt




def Possion_Solution(n):
    '''
    1.1 布丰投针问题
    We will skip the prove part which means direct mimic the possion process
    to compute approximate of `π`(Pi)

    In our experiment, 10^6 reach the res can reach np.isclose(sim_pi, pi) == True
    '''

    #define a class of needle flip
    class Needle_fp():
        def __init__(self, span, l):
            #the distance of half needle to parallels in a vertial direction
            self.x = None	#range(0, span/2)
            #the triangle of needle pose to parallels
            self.y = None	#range(0, pi/2)
            self.span = span
            self.l = l

        def flip(self):
            #for the variable fo x and y is a uniform distribution, we will
            #use uniform to sample their values
            self.x = random.uniform(0, self.span/2)
            self.y = random.uniform(0, pi/2)
            return (self.x, self.y)

    #Frist, define a needle flip
    span, l = 10, 9.5
    processor = Needle_fp(span, l)
    
    record = [processor.flip() for _ in range(n)]
    
    #valid record: x < l/2*sin(y), use count to compute the valide probability
    Prob = len([rec for rec in record if rec[0] < (l/2)*np.sin(rec[1])])/n

    #compute pi
    # the prob is 2*l/(pi*span)
    sim_pi = ((2*l)/Prob)/span
    print(sim_pi)
    return np.isclose(sim_pi, pi)



#print("Whether the `pi` we computed is closed to math.pi?: {}".format(Possion_Solution(10000000)))



'''
1.2 蒙特卡洛积分
'''

'''
1.3  蒙特卡洛采样
'''
# - Inverse Sampling
def sampleExp(Lambda = 2,maxCnt = 50000):
    ys = []
    standardXaxis = []
    standardExp = []
    for i in range(maxCnt):
        u = np.random.random()
        y = -1/Lambda*np.log(1-u) #F-1(X) the inverse function
        ys.append(y)
    for i in range(1000):
        t = Lambda * np.exp(-Lambda*i/100)
        standardXaxis.append(i/100)
        standardExp.append(t)
    plt.plot(standardXaxis,standardExp,'r')
    plt.hist(ys,1000,normed=True)
    plt.show()


# reject sampling
def Importance_sampling(k=3, maxCnt = 50000):
    #let's try to sample a exp distribution with reject sampling
    
    #define simpler sampling
    def g():
        #we want a distribution close to exp, the Taylor expansion of e^x is nearly equal to 1 + x ...
        #let define this distribution as 2/3(x+1)
        return np.exp(np.random.uniform()) / (2/3 * (1+np.random.uniform()))

    samples = []
    while len(samples) < maxCnt:
        samples.append(g())
        #alpha = np.exp(np.random.uniform()) / u
        #if u <= alpha:
        #    samples.append(u)

    plt.hist(samples, bins = 10)
    plt.show()
    #We will see the plot is close to exp distrbution

Importance_sampling(maxCnt = 10000)




'''
3.1 MCMC 算法
- MH 算法
'''



def Gibbs_sampling():
    '''
    3.2 Gibbs 算法
    We want to sample a target sequence follow distribution of {0.5, 0.5}
    from a mulitinominal distribution of {p1->list, p2->list}(see load_toy_data)
    '''
    
    def load_toy_data():
        global p1, p2
        p1 = np.array([randint(0, 1000) for _ in range(100)])
        p1 = [p1[i]/p1.sum() for i in range(100)]
        p2 = np.array([randint(0, 500) for _ in range(50)])
        p2 = [p2[i]/p2.sum() for i in range(50)]

        data = [(np.random.multinomial(1, p1).argmax(), np.random.multinomial(1, p2).argmax())\
             for _ in range(100)]
        return np.array(data)

    def Prior(label):
        cnt = Counter(label)

        prior = {}
        for k in cnt.keys():
            prior[k] = cnt[k] / len(label)
        return prior

    def condition_Prob(cond_var, prior_var):
        '''Compute the condition probability of cond_var vs prior variable'''
        #get value and index of condition and prior
        prior_idx, prior_val = prior_var
        cond_idx, cond_val = cond_var

        #prior probability
        prior_prob = feature_priors[prior_idx][prior_val]

        #get condition samples
        samples = [sample[cond_idx] for sample in data if sample[prior_idx] == prior_val]

        #condition probability
        cond_Prob = Prior(samples)[cond_val] * prior_prob 
        return round(cond_Prob, 5)


    def construct_transfer_matrix(samples):
        n = len(samples)
        transfer_Matrix = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                sample_idx = (samples[i][0] - samples[j][0])/np.mean([samples[i][0], samples[j][0]]) + \
                    (samples[i][1] - samples[j][1])/np.mean([samples[i][1], samples[j][1]])
                sample_idx = 1 - sample_idx/2

                if i == j:
                    transfer_Matrix[i][j] = float('-inf')
                elif samples[i][0] == samples[j][0]:
                    transfer_Matrix[i][j] = condition_Prob((1, samples[j][1]), (0, samples[i][0])) * sample_idx
                elif samples[i][1] == samples[j][1]:
                    transfer_Matrix[i][j] = condition_Prob((0, samples[j][0]), (1, samples[i][1])) * sample_idx
                else:
                    transfer_Matrix[i][j] = 0.

        return transfer_Matrix

    global data
    data = load_toy_data()
    n_feature = data.shape[1]
    
    global feature_priors
    feature_priors = [Prior(data[:, i]) for i in range(n_feature)]

    trans_matrix_Q = construct_transfer_matrix(data)
    #for x in trans_matrix_Q: print(x)

    #iteration the detailed balance condition
    #sample 10 instance as seqence

    seq_idx = [random.randint(0, 99) for _ in range(10)]
    seq = np.array([data[idx] for idx in seq_idx])

    iters = 100
    while iters:
        for i in range(len(seq)):
            seq_idx[i] = np.argmax(trans_matrix_Q[seq_idx[i]])
            seq[i] = data[seq_idx[i]]

        print([tuple(x) for x in list(seq)])
        iters -= 1
    print("We can see the sample transfer state of variable-1 and variable-2 stick to the distribution of [0.5, 0.5]\
        . While the Source data follow the multibinominal of [p1 = {},\n p2 ={}]".format(p1, p2))


#Gibbs_sampling()