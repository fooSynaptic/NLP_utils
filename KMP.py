#py3
import time


def common_len(pre, suff):
    m = 0
    for i in pre:
        for j in suff:
            if i == j: m = max(m, len(i))
    
    return m



def KMP(S, p):
    sub_p = [p[:i] for i in range(1, len(p))]
    sub_p.append(p)

    move_next = [0 for _ in range(len(p))]

    for i in range(1, len(p)-1):
        #print(sub_p[i])
        prefix, suffix = [sub_p[i][:k] for k in range(1, len(sub_p[i]))], \
            [sub_p[i][-k:] for k in range(1, len(sub_p[i]))]
        #print(prefix, suffix)
        move_next[i] = common_len(prefix, suffix)

    #modify the phrase
    move_next.pop()
    move_next.insert(0, -1)


    #start mapping
    i, j = 0, 0
    while i < len(S) and j < len(p):
        print(S)
        print(' '*i + p)
        print(' '*i + '^')

        if S[i] == p[j] or j == -1:
            print('incre i and j')
            i += 1
            j += 1
            print('after incre', i, j)
        else:
            print("Move next...")
            j = move_next[j]
            print(i, j)
        
        if j == len(p):
            print('The mapping stirng of S refer to p start from ', i-j)
            return i-j
    print("No mapping...")
    return -1






        
S = 'acababaabcacabc'
p = 'abaabcac'

KMP(S, p)