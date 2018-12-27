import numpy as np
import random
 
def gen_inv(a):
	a_sq = np.dot(a.T, a)
	eigen = np.linalg.eig(a_sq)
	eigen_vals = eigen[0]
	eigen_vectors = eigen[1]
 
	orth = a.dot(eigen_vectors)
	new_orth_len = np.zeros([orth.shape[1], orth.shape[1]])
	orth_sq = orth.T.dot(orth)

	for j in range(orth.shape[1]):
		for i in range(orth.shape[0]):
			orth[i][j] /= (orth_sq[j][j] ** 0.5)
		new_orth_len[j][j] = orth_sq[j][j] ** 0.5

	return {"Q": orth, "lamda": new_orth_len, "P": eigen_vectors}
 
M = np.zeros([500, 500])
for i in range(M.shape[0]):
	for j in range(M.shape[1]):
		M[i][j] = random.random()
 
print(gen_inv(M))


