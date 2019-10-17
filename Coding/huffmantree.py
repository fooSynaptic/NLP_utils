from collections import Counter

'''
class huffman_tree(object):
	def __init__(self, right, left):
		#assert isinstance(root, right, left)
		self.root = left + right
		self.left = left
		self.right = right
		self._node_print()

	def _add_node(self, numset):
		assert isinstance(numset, list)
		numset.sort()
		numset = [x for x in numset if x>max(self.right,self.left)]
		for i in numset:
			self._update_root(i)

	def _update_root(self, merger):
		#self._node_print()
		self.right = self.root
		self.root = self.root + merger
		self.left = merger
		#print('{}<({},{})'.format(self.root, self.right, self.left))
		self._node_print()
	
	def _node_print(self):
		print('{}<({},{})'.format(self.root, self.right, self.left))


if __name__ == "__main__":
	tree = huffman_tree(2,5)
	tree._add_node([7, 13])
'''


class node():
	def __init__(self, val):
		self.val = val
		self.left = None
		self.right = None


vec = [1,3,5,6,8,15]
def construct(code_vec):
	code_vec = sorted(code_vec)

	#tree initialize
	right, left = code_vec[:2]
	head = sum(code_vec[:2])
	root = node(head)
	root.right = node(right)
	root.left = node(left)

	remain_vec = code_vec[2:]
	while len(remain_vec):		
		if remain_vec[0] < root.val:
			val_candidate = remain_vec.pop(0)
			new_root = node(val_candidate + root.val)
			new_root.left = (node(val_candidate) if val_candidate > root.val else root)
			new_root.right = (node(val_candidate) if val_candidate < root.val else root)
			root = new_root
		elif not len(remain_vec): break
		elif remain_vec[0] > root.val:
			val_candidate = remain_vec.pop(0)
			additive_val = remain_vec.pop(0)
			additive_root = node(val_candidate + additive_val)
			additive_root.left = node(max(val_candidate, additive_val))
			additive_root.right = node(min(val_candidate, additive_val))
			higher_root = node(root.val + additive_root.val)
			if root.val > additive_root.val:
				higher_root.left, higher_root.right = root, additive_root
			else:
				higher_root.left, higher_root.right = additive_root, root
			root = higher_root
		print("tmp root:", root.val)

	return root



def inorder_traverse(head):
	if not head: return
	print(head.val)
	inorder_traverse(head.left)
	inorder_traverse(head.right)

inorder_traverse(construct(vec))