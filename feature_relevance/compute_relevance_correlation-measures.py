from __future__ import division
from scipy import histogram, digitize, stats, mean
from collections import defaultdict

import sys
import pandas as pd
import scipy as sc
import numpy as np

from math import log
log2 = lambda x:log(x,2)


root = sys.argv[1] #Data directory

def mutual_information(x,y):
    ce, hx = conditional_entropy(x,y)
    hy = entropy(y)
    return hy - ce

def norm_mutual_information(x,y):
    ce, hx = conditional_entropy(x,y)
    hy = entropy(y)
    return (hy - ce)/sc.sqrt(hx*hy)
    
def conditional_entropy(x, y):
    
    """
    x: real numbers vector
    y: integer vector
    compute H(Y|X)
    """
    # discretize X
    
    hx, bx = histogram(x, bins=x.size/10, density=True) #this discretization should be improved

    Py = compute_distribution(y)
    
    Px = compute_distribution(digitize(x,bx))
    
    res = 0
    for ey in set(y):
        # P(X | Y)
        x1 = x[y==ey]
        condPxy = compute_distribution(digitize(x1,bx))

        for k, v in condPxy.iteritems():
            res += (v*Py[ey]*(log2(Px[k]) - log2(v*Py[ey])))
    
    en_x =entropy(digitize(x,bx))
    
    return res, en_x
        
def entropy(y):
    
    """
    Calcola l'entropia di un vettore di discreti
    """

    # P(Y)
    Py = compute_distribution(y)
    res = 0.0
    for k, v in Py.iteritems():
        res += v*log2(v)
    return -res

def compute_distribution(v):
    
    """
    v: vettore di interi
    
    ottengo un dictionary con chiave pari all'intero e valore pari alla probabilita
    """

    d = defaultdict(int)
    for e in v: d[e] += 1
    
    s = float(sum(d.values()))

    return dict((k, v/s) for k, v in d.items())

print "Reading from ", root

df = pd.read_csv(root + r'output/sample_train_features.txt', sep='\t',header=0)
print len(df), " rows"

max_feature_id = sys.argv[2].strip()

Y = df['label']
X = df.loc[:,'feat1':'feat' + max_feature_id] # il numero di features dipende dal dataset

names = list(X.columns.values)

Y0 = Y.as_matrix()
norm_miscore = []
kendall_score = []
spearman_score = []

for i in range(len(names)):
    print "Spear,NMI,Ken", i
    
    X0 = X.loc[:,names[i]].as_matrix()

    spearman_score.append(stats.spearmanr(X0,Y0)[0])
    norm_miscore.append(norm_mutual_information(X0, Y0))
    kendall_score.append(stats.kendalltau(X0,Y0)[0])

AGvar = []

#calcolo i gruppi

nrs = set(Y0)
idx = []
for i in range(len(nrs)):
    idx.append(Y[Y == float(i)].index.tolist())

for i in range(len(names)):
    print "ag", i
    N = len(X.loc[:,names[i]].as_matrix())
    
    feature_mean = mean(X.loc[:,names[i]].as_matrix())
    
    TSS = sum([(xx-feature_mean)**2 for xx in X.loc[:,names[i]].as_matrix()])
    SSDX = 0    

    for j in range(len(nrs)):
        Ng = len(X.loc[idx[j],names[i]].as_matrix())
        # variation between groups (X for 'cross')
        SSDX += Ng*(mean(X.loc[idx[j],names[i]].as_matrix())-feature_mean)**2
    
    AGvar.append(1-SSDX/TSS)
    
with open(root+r'output/feature_rank.txt','w') as wfile:
    wfile.write("label\tNMI\tAG1\tKen\tSpea\n")
    for i in range(len(names)):
        wfile.write("%s\n" % (str(names[i]) +
                              "\t"+str(norm_miscore[i]) +
                              "\t"+str(-log(AGvar[i])) +
                              "\t"+ str(np.abs(kendall_score[i])) +
                              "\t"+ str(np.abs(spearman_score[i]))
                              )
                    )
wfile.close()