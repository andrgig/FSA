from __future__ import division
from scipy import stats
import pandas as pd
import sys

#data root
root = sys.argv[1]

#read train data
df = pd.read_csv(root + r'output/sample_train_features.txt', sep = '\t', header = 0)

#Let Y be the label vector
Y = df['label']

#Let X be the feature matrix
X = df.drop(["label", "query"], axis = 1)
#X = df.loc[:,'feat1':'feat220']

#reset df to free space
df = 0

#get a list of features names
names = list(X.columns.values)

spear_corr = []

with open(root + r'output/spear_corr.txt', 'w') as f:
    f.seek(0)
    f.truncate()
f.close()

for i in range(len(names)):
    
    temp = []
    
    #compute spearman for each pair of feature
    
    for j in range(len(names)):
        print "i,j: ",i,j
        if j > i:
            Xi = X.loc[:,names[i]].as_matrix()
            Xj = X.loc[:,names[j]].as_matrix()
            temp.append(stats.spearmanr(Xi,Xj)[0])
            
        elif j == i:
            temp.append(1.0)
        else:
            temp.append(spear_corr[j][i])
    
    #save spearman matrix in a tab separated file
    
    with open(root + r'output/spear_corr.txt', 'a') as f:
        for ele in temp:
            f.write("%f\t" % ele)
        f.write("\n")

    spear_corr.append(temp)