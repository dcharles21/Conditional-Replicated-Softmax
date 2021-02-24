# Analysis Code

# Load Libraries
import numpy as np
from preprocess import num_authors, data_train, data_test, vocab, Z_train, Z_test
from cRSM import v, Zv, w, wm, a, b, logZ, num_test_docs

# Average Test Perplexity
D = v.sum(axis = 1)
num_words = D.sum()  

Wh = np.dot(v, w) + + np.dot(Zv, wm) + np.outer(D,a)
expWh = np.exp(Wh)
bias_v = np.dot(v,b)

log_pv = np.log(1 + expWh).sum(axis = 1) + bias_v - logZ
log_pv = log_pv/D
np.exp(-log_pv.sum()/num_test_docs)

# Average Test Perplexity for each  heldout documents
np.exp(-log_pv)

# Penalized Perplexity
K = len(vocab)
F = 50
M = num_authors
p =  K*F + M*F + K + F

D = v.sum(axis = 1)
num_words = D.sum()  

Wh = np.dot(v, w) + + np.dot(Zv, wm) + np.outer(D,a)
expWh = np.exp(Wh)
bias_v = np.dot(v,b)

log_pv = np.log(1 + expWh).sum(axis = 1) + bias_v - logZ
np.exp((-log_pv.sum() + p)/num_words)

## Five Number Summary ##

# Average Test Perplexity for each  heldout documents
perps = np.exp(-log_pv)

# Calculate Quartiles
quartiles = np.percentile(perps, [25, 50, 75])

# Calculate min/max
data_min, data_max = perps.min(), perps.max()

# print 5-number summary
print('Min: %.3f' % data_min)
print('Q1: %.3f' % quartiles[0])
print('Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Max: %.3f' % data_max)
