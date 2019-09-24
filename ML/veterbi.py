# encoding = utf-8
# /usr/bin/python3


states = ('Rainy', 'Sunny')
 
observations = ('walk', 'shop', 'clean')
 
start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
 
transition_probability = {
    'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
    'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
    }
 
emission_probability = {
    'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
}
 
# 打印路径概率表
def print_dptable(V):
    print("    "),
    print('\t', '\t'.join([str(i) for i in range(len(V))]))
 
    for y in V[0].keys():

        print(y+'\t', '\t'.join([str(round(V[t][y], 3)) for t in range(len(V))]))
        #print('\t')
 
 
def viterbi(obs, states, start_p, trans_p, emit_p):
    """
 
    :param obs:观测序列
    :param states:隐状态
    :param start_p:初始概率（隐状态）
    :param trans_p:转移概率（隐状态）
    :param emit_p: 发射概率 （隐状态表现为显状态的概率）
    :return:
    """
    # 路径概率表 V[时间][隐状态] = 概率
    V = [{}]
    # 一个中间变量，代表当前状态是哪个隐状态
    path = {}
 
    # 初始化初始状态 (t == 0)
    for y in states:
        #初始状态概率等于初始状态乘以发射概率
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
    print('Initial state:', V, path)
 
    # 对 t > 0 跑一遍维特比算法
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
 
        for y in states:
            # 概率 隐状态 =    前状态是y0的概率 * y0转移到y的概率 * y表现为当前状态的概率
            (prob, state) = max([(V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states])
            # 记录最大概率
            V[t][y] = round(prob, 3)
            # 记录路径
            newpath[y] = path[state] + [y]
 
        # 不需要保留旧路径
        path = newpath
        print('steps-', t,'\t', V, path)
 
    print_dptable(V)
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return (prob, path[state])
 
 
def example():
    return viterbi(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)
 
 
print(example())