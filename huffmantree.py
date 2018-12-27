from collections import Counter


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
