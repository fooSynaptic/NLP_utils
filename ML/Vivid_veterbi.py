# encoding = utf-8
# /usr/bin/python3

'''
This program implemented the vertibi algorithm inference while have 
- 1. The state-state transition matrix
- 2. The state-observation transition matrix
- 3. Initial state

- ref: https://zhuanlan.zhihu.com/p/63087935
'''

__author__ = 'fooSynaptic'

import numpy as np



def verterbi(initial_state, s_s_mat, s_o_mat):
    assert len(initial_state) == len(s_s_mat) == len(s_o_mat[0]) and len(s_s_mat) == len(s_s_mat[0]), \
        print("Input Dimension Err.")

    state_num, observe_num = np.shape(s_o_mat)

    #Initial mat1 and mat2
    #mat1_r, mat1_c = []
    mat1 = [[None for _ in range(observe_num)] for _ in range(state_num)]
    mat2 = [[None for _ in range(observe_num)] for _ in range(state_num)]

    for i in range(state_num):
        mat1[i][0] = initial_state[i] * s_o_mat[i][0]
        mat2[i][0] = 0

    for t in range(1, observe_num):
        for s in range(state_num):
            state_trans = [mat1[s_pre][t-1] * s_s_mat[s_pre][s] for s_pre in range(state_num)]
            mat1[s][t] = max([s_trans * s_o_mat[s][t] for s_trans in state_trans])
            mat2[s][t] = np.argmax(state_trans)

    
    ### decode
    #paths = [np.argmax([mat1[i][t] for i in range(state_num)]) for t in range(state_num-1, -1, -1)]
    paths = []
    lst_state = np.argmax([mat1[i][observe_num-1] for i in range(state_num)])
    paths.append(lst_state)

    for t in range(observe_num-2, 0, -1):
        lst_state = mat2[lst_state][t]
        paths.append(lst_state)
    
    paths.append(mat2[lst_state][1])


    return paths


def test_case():
    init_state = [0.5, 0.1, 0.4]
    Trans = [
        [0.1, 0.2, 0.7],
        [0.2, 0.3, 0.5],
        [0.3, 0.4, 0.3]
    ]

    Emis = [
        [0.2, 0.2, 0.6],
        [0.3, 0.3, 0.4],
        [0.4, 0.4, 0.2]
    ]

    print(verterbi(init_state, Trans, Emis))


test_case()