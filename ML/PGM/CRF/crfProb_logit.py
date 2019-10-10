# encoding = utf-8
# /usr/bin/python3

"""
Solve the probability computation Problem in CRF
Input: CRF features and params
Output: Condition probability of state sequence given observation
"""
import numpy as np



def transFunc(yPre, yi, x, i, lambdaIndex):
    if lambdaIndex == 0:
        return (1 if yPre == 1 and yi == 2 and i in [2, 3] else 0)
    elif lambdaIndex == 1:
        return (1 if yPre == 1 and yi == 1 and i == 2 else 0)
    elif lambdaIndex == 2:
        return (1 if yPre == 2 and yi == 1 and i == 3 else 0)
    elif lambdaIndex == 3:
        return (1 if yPre == 2 and yi == 1 and i == 2 else 0)
    elif lambdaIndex == 4:
        return (1 if yPre == 2 and yi == 2 and i == 3 else 0)


def stateFunc(yi, x, i, miuIndex):
    if miuIndex == 0:
        return (1 if yi == 1 and i == 1 else 0)
    elif miuIndex == 1:
        return (1 if yi == 2 and i in [1, 2] else 0)
    elif miuIndex == 2:
        return (1 if yi == 1 and i in [2, 3] else 0)
    elif miuIndex == 3:
        return (1 if yi == 2 and i == 3 else 0)
        




def logit(observeTimes, statePath):
    transW = [1, 0.6, 1, 1, 0.2]
    stateW = [1, 0.5, 0.8, 0.5]

    Transfeature, statePath_feature = 0, 0
    for i in range(len(transW)):
        for step in range(1, observeTimes):
            Transfeature += transW[i] * transFunc(statePath[step-1], statePath[step], None, step+1, i)

    for j in range(len(stateW)):
        for step in range(0, observeTimes):
            statePath_feature += stateW[j] * stateFunc(statePath[step], None, step+1, j)

    
    return np.exp(Transfeature + statePath_feature)





print("According to the reference of 《统计学习方法》by 李航, reference answer is near exp(3.2)")
print("Is this Solution right? ", np.isclose(logit(3, [1, 2, 2]), np.exp(3.2)))