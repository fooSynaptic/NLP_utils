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
        self.left = None
        self.right = None

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


def cluster_viewer(node, lv, tree):
    #print(lv , tree)
    if node:
        print('\t'*lv, node.get_root())
        if lv > len(tree)-1:
            tree.append([node.get_root()])
        else:
            tree[lv].append(node.get_root())
        if not node.get_left() is False:
            cluster_viewer(node.left, lv+1, tree)
        if not node.get_right() is False:
            cluster_viewer(node.right, lv+1, tree)  

    
def KD_generation(node):
    #print('Root', node.get_root())
    #print('right:', node.get_right(),'left', node.get_left())
    if not node.get_left() is False:
        node.left = cluster(node.get_left())
        node.left.partition()
        KD_generation(node.left)
    if not node.get_right() is False:
        node.right = cluster(node.get_right())
        node.right.partition()
        KD_generation(node.right)



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
    KD_generation(curr)

    #traverse root for visualization
    viewer = root
    Tree = []
    print("The levels denote the tree depth, same level means they stay in parallel,'\n'\
    and the next level denote the parent and child information, in our code,'\n \
    for two stacked node, right node first and left node second...")
    print(''.join(' level{}'.format(i)+'\t' for i in range(cluster_max)))
    cluster_viewer(viewer, 0, Tree)
    


if __name__ == '__main__':
    construct(test_data)