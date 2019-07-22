#py3
from random import randint


train_table = [[None for _ in range(100)] for _ in range(3)]

for i in range(100):
    train_table[0][i] = randint(1, 5)
    train_table[1][i] = ("S" if randint(0, 1) == 0 else 'M')
    train_table[2][i] = (1 if randint(0, 1) == 0 else -1)

for x in train_table: print(x)


def naive_bayes_classfication(data, f1, f2):
    '''
    Classfy a sample with feature_0, feature_1 as 2, S respectivey, 
    try to determine the value of its feature_3

    We want to build n-1 look-up tables which store the Conditional probabilites.
    '''
    sample_num, feature_num = len(data[0]), len(data)

    #first we compute the prior-probability refer to the predict feature- feature_3.
    prior = {}
    for i in range(sample_num):
        if data[2][i] in prior:
            prior[data[2][i]] += [i]
        else:
            prior[data[2][i]] = [i]
    
    prior_prob = {}
    for k in prior.keys():
        prior_prob[k] = len(prior[k])/sample_num


    #build the conditional prob tables
    print('prior', prior)
    cond_table = {}
    n_prior = len(prior_prob.keys())

    for i in range(feature_num-1):
        cond_table[i] = {}
        for k in prior.keys():
            cond_table[i][k] = {}

            for ins in prior[k]:
                if data[i][ins] in cond_table[i][k]:
                    cond_table[i][k][data[i][ins]] += [ins]
                else:
                    cond_table[i][k][data[i][ins]] = [ins]

    
    #deduction
    res = {}
    for p in prior_prob.keys():
        print("If we start from state of {}".format(p))
        posterior = prior_prob[p] *\
            len(cond_table[0][p][f1])/len(prior[p]) * len(cond_table[1][p][f2])/len(prior[p])
        print('The condition when {} -> {} -> {} happend with prob of {}'.\
            format(p, f1, f2, round(posterior, 2)))
        res[p] = posterior

    ans = sorted(res.items(), key=lambda res:res[1])[-1][0]
    print("The most likely feature of fearure_3 with f1 and f2 is {}".format(ans))




bayes_classfication(train_table, 2, 'S')



