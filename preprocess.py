# Preprocess Documents

# Load Libraries
import csv
import numpy as np
import scipy.io as sio

# Load Documents
paper = []
filename = "MAT_NIPS.txt"
with open(filename) as csvfile:
    readCSV = csv.reader(csvfile, delimiter = ',')

    for row in readCSV:
        print(row[0])
        paper.append(row[0])

#List of Authors
authors = sio.loadmat('authors_nips.mat')
num_authors = len(authors['AN']) # 2037 authors

author = []
for i in range(0, num_authors):
    author.append(authors['AN'][i][0][0])

#Preprocess Training Data
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=0.95, min_df=2, max_features = 7500)

data = cv.fit_transform(paper)
data = data.toarray()

# Change word count to log(1 + w_i)
Data = np.zeros((len(data), 7500))
Data = np.log(1 + data)
Data = np.around(Data)

# Delete any 0 vectors in dataset
Data = Data[~np.all(data == 0, axis=1)]

# Shuffle data vector
# np.random.shuffle(Data)

data_train = Data[0:1690].astype(int)
data_test = Data[1690:1740].astype(int)

# Vocabulary
vocab = cv.get_feature_names()

#Load Author Vector
author_counts = sio.loadmat('authordoc_nips.mat')
author_counts['AD']

aut_doc = np.zeros((num_authors, 1740))
for i in range(0, num_authors):

    index = np.nonzero(author_counts['AD'][i].todense())
    index = index[1] # which papers did the ith author write

    for j in index:
        aut_doc[i][j] = 1
z = aut_doc.T # author vector

Z_train = z[0:1690]
Z_test = z[1690:1740] 

