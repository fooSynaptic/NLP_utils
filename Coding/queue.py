#py3

class MyCircularQueue(object):

    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        :type k: int
        """
        self.size = k
        self.queue = [None for _ in range(k)]
        self.head, self.tail = -1, -1
        
    def enQueue(self, value):
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        if not self.isFull():
            if self.tail < 0 and self.head < 0:
                self.tail, self.head = 0, 0
            elif self.head == 0 and self.queue[0] != None:
                self.tail += 1
            elif self.tail == self.size-1:
                self.tail = 0

            self.queue[self.tail] = value
            return True
        else:
            #print(self.queue)
            #print([x for x in self.queue if x])
            return False
            
        

    def deQueue(self):
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        :rtype: bool
        """
        if not self.isEmpty():
            if self.head == self.size-1:
                self.queue[self.head] = None
                self.head = 0
            else:
                self.queue[self.head] = None
                self.head += 1
            return True
        else:
            return False

        

    def Front(self):
        """
        Get the front item from the queue.
        :rtype: int
        """
        if self.isEmpty():
            return -1
        
        for x in self.queue:
            if x:
                return x

    def Rear(self):
        """
        Get the last item from the queue.
        :rtype: int
        """
        if self.isEmpty():
            return -1
        
        for x in self.queue[::-1]:
            if x: return x
        

    def isEmpty(self):
        """
        Checks whether the circular queue is empty or not.
        :rtype: bool
        """
        return len([x for x in self.queue if x]) == 0

    def isFull(self):
        """
        Checks whether the circular queue is full or not.
        :rtype: bool
        """
        return len([x for x in self.queue if x]) == self.size


#["MyCircularQueue","enQueue","enQueue","enQueue","enQueue","Rear","isFull","deQueue","enQueue","Rear"]
#[[3],[1],[2],[3],[4],[],[],[],[4],[]]

obj = MyCircularQueue(3)
param_1 = obj.enQueue(1)
print(obj.tail, obj.queue)
param_2 = obj.enQueue(2)
print(obj.tail, obj.queue)
param_3 = obj.enQueue(3)
print(obj.tail, obj.queue)
param_4 = obj.enQueue(4)
print(obj.tail, obj.queue)
param_5 = obj.Rear()
print(obj.tail, obj.queue)
param_6 = obj.isFull()
print(obj.tail, obj.queue)

param_7 = obj.deQueue()
print(obj.tail, obj.queue)

param_8 = obj.enQueue(4)
print(obj.tail, obj.queue)
param_9 = obj.Rear()
print(obj.tail, obj.queue)

for i in dir():
    if 'param' in i:
        print(i, eval(i))