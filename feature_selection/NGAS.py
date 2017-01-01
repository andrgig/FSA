'''
Description:
GREEDY ALGORITH: return the L features most correlated with y and less correlated with the other features. starting with the feature 
most correlated with y

Input:
A similarity matrix and a relevance vector from files

Output:
A list of feature subsets
'''

import sys

#file paths
root = sys.argv[1]
similarity_path = root + 'output/spear_corr.txt'
relevance_path = root + 'output/NDCG_single_feature.txt'


#set the number of feature to select correstponding to 5%, 10%, 20%, 30%, 40%, 50%,75% of the feature set
L=[26,52,104,156,208,260,389]

#list of features to be excluded

blacklist_file = open(root+'output/blacklist.txt','r')

blacklist_row = blacklist_file.readline()

blacklist = []

while (blacklist_row != ''):
    blacklist_row_fields = blacklist_row.strip().split('\t')
    blacklist.append(blacklist_row_fields[0])
    blacklist_row = blacklist_file.readline()

lists=[]

#feature list will contatin the feature number as int
#e.g. [99, 1, 2, 11, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 12, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 13, 121, 122, 123, 124]

for nr in L:
    
    #read similarity matrix
    corr_file = open(similarity_path,'r')
    
    #build a dictionary of dictionaries containing correlation matrix
    '''
    D={feat1: {feat1: corr(x1,x1), feat2: corr(x1,x2),...featn:corr(x1,xn)},
       feat2: {feat1: corr(x2,x1), feat2: corr(x2,x2),...featn:corr(x2,xn)},
       ...
       featn: {feat1: corr(x2,x1), feat2: corr(x2,x2),...featn:corr(x2,xn)}}
    '''
    
    D = dict()
    
    corr_row = corr_file.readline()
    corr_row = corr_row.strip()
    corr_row_fields = corr_row.split('\t')
    
    for i in range(1,len(corr_row_fields)+1):
            feat = "feat" + str(i)
            if feat not in blacklist:
                D[feat] = dict()
    j = 1
    
    while (corr_row != ""):
        
        corr_row = corr_row.strip()
        corr_row_fields = corr_row.split('\t')
        
        for i in range(0,len(corr_row_fields)):
            
            feat1 = "feat" + str(j)
            feat2 = "feat" + str(i+1)
            
            if (feat1 not in blacklist) and (feat2 not in blacklist):
                if feat1 != feat2:
                    D[feat1][feat2] = abs(float(corr_row_fields[i]))
                    
        j += 1
        
        corr_row = corr_file.readline()
        
    #upload the relevance vector from the file
    r_file = open(relevance_path,'r')
    
    #build a dictionary containing relevance(feature(i),y)
    R = dict()
    
    #uncommentif you want to skip headline
    #r_row=r_file.readline()
    
    r_row = r_file.readline()
    
    r_row = r_row.strip()
    #r_row_fields=r_row.split('\t')
    
    j = 1
    
    while (r_row != ""):

        feat = "feat" + str(j)
        r_row = r_row.strip()
        #r_row_fields=r_row.split('\t')
        
        if feat not in blacklist:
            R[feat] = abs(float(r_row))

        j += 1

        r_row = r_file.readline()

    #GREEDY ALGORITH: return the L features most "correlated" with y 
    #and less "correlated" with the others, starting with the feature 
    #mostly correlated with y
    
    #initialize feature subset list
    feature_list = []
    
    #start from the most y-correlated feature
    temp = max(R, key = R.get)
    
    feature_list.append(int(temp.strip('feat')))
    
    #remove selected from Relevance Vector
    R.pop(temp,0)
        
    while len(feature_list) < nr:
        
        #find the feature dictionary with minimum similarity with temp
        fmin = min(D[temp], key = D[temp].get)
        
        #remove selected from dictionay of dictionaries
        for d in D:
            D[d].pop(temp, 0)
        D.pop(temp, 0)
        
        #find feature most similar to fmin
        fmax = max(D[fmin], key = D[fmin].get)
        
        #select the most relevant feature between fmin and fmax
        if R[fmin] >= R[fmax]:
            feature_list.append(int(fmin.strip('feat')))
            temp = fmin

        else:
            feature_list.append(int(fmax.strip('feat')))
            temp = fmax

        R.pop(temp, 0)
    
    lists.append(feature_list)

print lists

with open(root + 'output/ngas_selection.txt', "w") as outfile:
    for list_ele in lists:
        outfile.write(' '.join([str(ele) for ele in list_ele]) + "\n")
