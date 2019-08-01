#py3
from collections import Counter
import numpy as np

'''
Build a KD tree for cluster algorithm..
reference: 
    - http://www.huaxiaozhuan.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0/chapters/5_knn.html
'''

# we use two dimension features for our test data
test_data = [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]


#KD-tree is a kind of binary search Tree, So we define BST first
class cluster:
    def __init__(self, data):
        self.data = data
        self.part_dim = np.argmax([np.var(data[:, i]) for i in range(data.shape[1])])
        self.part_median = np.median(data[:, self.part_dim])
        self.left_nodes = []
        self.right_nodes = []
        self.root_nodes = []

    def partition(self):
        for i in range(self.data.shape[0]):
            if self.data[i][self.part_dim] == self.part_median:
                self.root_nodes.append(list(self.data[i]))
            elif self.data[i][self.part_dim] > self.part_median:
                self.right_nodes.append(list(self.data[i]))
            else:
                self.left_nodes.append(list(self.data[i]))


    def get_left(self):
        if not self.left_nodes: return False
        #print('left:', self.left_nodes)
        return np.array(self.left_nodes)


    def get_right(self):
        if not self.right_nodes: return False
        #print('right:', self.right_nodes)
        return np.array(self.right_nodes)


    def get_root(self):
        if not self.root_nodes: return False
        return np.array(self.root_nodes)


def construct(data):
    #get the dimension of raw data
    n_features = Counter([len(x) for x in data]).most_common(1)[0][0]

    #select the dimension with the largest variable
    features = np.array([[x[i] for i in range(n_features)] for x in data])

    #root_dim = np.argmax([np.var(features[:, i]) for i in range(n_features)])
    root = cluster(features)
    root.partition()
    cluster_max = len(data)
    cluster_cnt = 0
    
    curr = root
    while curr.get_left() is not False and cluster_cnt < cluster_max:
        print('partition left:left', curr.get_left(), '\n', \
            'partition left:right', curr.get_right())
        sub_cluster = cluster(curr.get_left())
        sub_cluster.partition()
        curr = sub_cluster
        cluster_cnt += 1
    #print('partition left:right', curr.get_right())

    curr = root
    while curr.get_right() is not False and cluster_cnt < cluster_max:
        print('partition right:left', curr.get_left(), '\n', \
            'partition right:right', curr.get_right())
        sub_cluster = cluster(curr.get_right())
        sub_cluster.partition()
        curr = sub_cluster
        cluster_cnt += 1
    #print(curr.get_left())


if __name__ == '__main__':
    construct(test_data)