class K_larget(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        if k == len(nums): return min(nums)
        self.k = k
        self.stack = [None for _ in range(k)]
        
        for num in nums:
            self.stack_maintain(num)

        return self.stack[0]
        
    
    def stack_maintain(self, n):
        if all([x == None for x in self.stack]):
            self.stack[-1] = n
            return
        elif any([x == None for x in self.stack]):
            idx = -1
            for i in range(self.k-1, -1, -1):
                if self.stack[i] == None:
                    break
                if n <= self.stack[i]:
                    idx = i
                else:
                    self.stack[i], n = n, self.stack[i]
                    idx = i
   
            self.stack.insert(idx, n)  
            self.stack.pop(0)
   
            return
                  
        else:
            if n > self.stack[-1]:
                self.stack = self.stack[1:] + [n]
                return
            elif n < self.stack[0]:
                return
            else:
                idx = 0
                for i in range(self.k):
                    if n <= self.stack[i]:
                        idx = i
                        break
                self.stack.insert(idx, n)
                self.stack.pop(0)
                return
                
                    
                
            
                
