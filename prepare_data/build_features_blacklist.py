import sys
import pandas as pd

root = sys.argv[1]

data = pd.read_csv(root + "output\\feature_rank.txt", sep="\t", header = 0 )
print 
data[data.isnull().any(axis = 1)][[0]].to_csv('output\\blacklist.txt', index = False, header = False)
data[data.NMI.notnull()][[0]].to_csv('output\\nonescludere.txt', index = False, header = False)