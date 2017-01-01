'''
Input: The script takes a relevance vector and a list of files. Each file contain the clustering of the feature set
accordingly to a cutting level.
Output: list of feature subsets
'''

import pandas as pd
import sys

root = sys.argv[1]

relevance_path = root + 'output/NDCG_single_feature.txt'

L=[26,52,104,156,208,260,389]

#list of features to be excluded

blacklist_file = open(root+'output/blacklist.txt','r')

blacklist_row = blacklist_file.readline()

blacklist = []

while (blacklist_row != ''):
    blacklist_row_fields = blacklist_row.strip().split('\t')
    blacklist.append(blacklist_row_fields[0])
    blacklist_row=blacklist_file.readline()

r_file = open(relevance_path, 'r')

#build a dictionary containing relevance(feature(i),y)
R = dict()

#uncomment the following to skip the header 
#r_row=r_file.readline() 

r_row = r_file.readline()

r_row = r_row.strip()
#r_row_fields=r_row.split('\t')

j = 1
lists = []

while (r_row != ""):
    
    feat = j
    r_row = r_row.strip()
    #r_row_fields=r_row.split('\t')
    
    if feat not in blacklist:
        R[feat] = float(r_row)
        
    j += 1
    r_row = r_file.readline()

for l in L:
    
    #leggo i file contenenti due colonne separate da virgola
    #la prima contiene il ref della feature la seconda il cluster di appartenenza
    
    '''
    ,x
    feature1,cluster(feature1)
    feature12,cluster(feature1)
    feature1,cluster(feature1)
    ...
    
    '''
    print root

    filename = root + "output/g_" + str(l) + ".txt"
    
    df = pd.read_csv(filename, sep=',', header = 0, names = ["mfeature", "mgroup"], skiprows = 0)

    sublist = []
    
    nr_to_select = l
    
    for i in range(1, nr_to_select+1):

        temp = df[df["mgroup"] == i]["mfeature"].as_matrix()

        a = {k: R[k] for k in temp}
        sublist.append(max(a, key = a.get))
        
    #print sublist
    lists.append(sublist)

print lists

with open(root+'output/hcas_selection.txt', "w") as outfile:
    for list_ele in lists:
        outfile.write(' '.join([str(ele) for ele in list_ele]) + "\n")
