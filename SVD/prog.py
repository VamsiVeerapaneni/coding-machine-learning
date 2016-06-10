import numpy
from matplotlib import pyplot as plt
la=numpy.linalg
words=['I', 'like', 'enjoy', 'deep', 'learning', 'NLP', 'flying', '.']
x=[[0, 2, 1, 0, 0, 0, 0, 0],
 [2, 0, 0, 1, 0, 1, 0, 0],
 [1, 0, 0, 0, 0, 0, 1, 0],
 [0, 1, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 1],
 [0, 1, 0, 0, 0, 0, 0, 1],
 [0, 0, 1, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 1, 1, 1, 0]]
u,s,v=la.svd(x,full_matrices=False)
plt.xlim([-1,1])
plt.ylim([-1,1])
for i in xrange(len(words)):
	plt.text(u[i,0],u[i,1],words[i])
plt.show()
