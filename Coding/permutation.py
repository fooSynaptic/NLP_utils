# source from leetcode
from time import time

xrange = range


def permuteUnique(nums, verbose = False):
    ans = [[]]
    for n in nums:
        new_ans = []
        for l in ans:
            for i in xrange(len(l)+1):
                new_ans.append(l[:i]+[n]+l[i:])
                if verbose: print("Before handle duplication: l:{}, i:{}, new_ans:{}".format(l,i,new_ans))
                if i<len(l) and l[i]==n: break              #handles duplication
        ans = new_ans
        if verbose: print("Updated ans:{}".format(ans))
    return ans











#print(permuteUnique([1,1,2]))


class Solution:
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        
        
        ans = [[]]
        for n in range(1, n+1):
            new_ans = []
            for l in ans:
                for i in range(len(l) + 1):
                    new_ans.append(l[:i] + [n] + l[i:])
                    if i < len(l) and l[i] ==n: break
                
            ans = new_ans
        
        ans = [''.join([str(x) for x in a]) for a in ans]
        z3ans = [int(x) for x in ans]
        ans = sorted(ans)
        #ans = [str(x) for x in ans]
        
        return ans[k - 1]


res = Solution()
t1 = time()
print(res.getPermutation(3,3))
t2 = time()
print("Step 1 eclapse: ", t2-t1)
print(res.getPermutation(9,24))
t3 = time()
print("Step 2 eclapse: ", t3-t2)
print(res.getPermutation(8,11483))
t4 = time()
print("Step 3 eclapse: ", t4-t3)


