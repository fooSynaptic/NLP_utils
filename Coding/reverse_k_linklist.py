#py3

class listnode():
    def __init__(self, val):
        self.val = val
        self.next = None



def buildlinklist():
    nodes = [i for i in range(1, 9)]
    root = listnode(nodes.pop(0))
    curr = root
    while nodes:
        curr.next = listnode(nodes.pop(0))
        curr = curr.next
    return root


def viewer(node):
    if not node:
        return
    res = []
    while node:
        res.append(node.val)
        node = node.next
    
    res = [str(x) for x in res]
    print('->'.join(res))
    return 0



def revese_linklist(head):
    if not head or not head.next:
        return head
    
    curr = revese_linklist(head.next)
    head.next.next = head
    head.next = None
    return curr

#定义从头开始的每隔三个node进行翻转
def reverse_k_cis(head, k):
    tmp = head
    for i in range(k-1):
        if tmp: tmp = tmp.next

    if tmp == None:
        return head

    head2 = tmp.next
    tmp.next = None

    newHead = revese_linklist(head)
    newHead2 = reverse_k_cis(head2, k)
    head.next = newHead2
    
    
    return newHead

#按照顺序从头开始每k个node进行翻转
print('按照顺序从头开始每k个node进行翻转...')
root = buildlinklist()
viewer(root)
newroot = reverse_k_cis(root, 3)
viewer(newroot)


'''
原题
给定一个单链表的头节点 head,实现一个调整单链表的函数，使得每K个节点之间为一组进行逆序，
并且从链表的尾部开始组起，头部剩余节点数量不够一组的不需要逆序。
输入：1->2->3->4->5->6->7->8
输出：1->2->5->4->3->8->7->6
step:
1:reverse all result 87654321
2:reverse cis for per k result 67834521
1: reverse all result 12543876
'''
root = buildlinklist()
print("raw link")
viewer(root)
reverse_step1 = revese_linklist(root)
print("step1 result link")
viewer(reverse_step1)
reverse_step2 = reverse_k_cis(reverse_step1, 3)
print("step2 result link")
viewer(reverse_step2)
reverse_step3 = revese_linklist(reverse_step2)
print("step3 result link")
viewer(reverse_step3)

