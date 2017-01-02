'''
The script reads a tab separated file representing features similarity and a file 
containing a relevance vector. Then it runs GAS in order to select the n 
most relevant features which are less similar with each other.

It starts from the feature most correlated with y.

Output: list of feature subsets containing the feature number as int
e.g. [99, 1, 2, 11, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 12, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 13, 121, 122, 123, 124]

'''
#################      INPUT      ##########################

import sys

#file path
root = sys.argv[1] 

#similarity matrix file path
similarity_path = root + 'output/spear_corr.txt'

#relevance vector file path
relevance_path = root + 'output/NDCG_single_feature.txt'

#hyperparameter
c = 0.01

#feature subsets to be produced, corresponding to 5%, 10%, 20%, 30%, 40%, 50%, 75% of the whole set
L = [11,22,44,66,88,110,165]

#list of features to be excluded

blacklist_file = open(root + 'output/blacklist.txt','r')

blacklist_row = blacklist_file.readline()

blacklist = []

while (blacklist_row != ''):
    blacklist_row_fields = blacklist_row.strip().split('\t')
    blacklist.append(blacklist_row_fields[0])
    blacklist_row = blacklist_file.readline()

#initialize a void list
lists = []

for nr_feat in L:
    
    print nr_feat
    
    corr_file = open(similarity_path,'r')
    
    #build a dictionary of dictionaries containing the correlation matrix
    
    '''
    D={feat1: {feat1: corr(x1,x1), feat2: corr(x1,x2),...featn:corr(x1,xn)},
       feat2: {feat1: corr(x2,x1), feat2: corr(x2,x2),...featn:corr(x2,xn)},
       ...
       featn: {feat1: corr(x2,x1), feat2: corr(x2,x2),...featn:corr(x2,xn)}}
    '''
    
    D = dict()
    
    corr_row = corr_file.readline().strip()
    
    #the similarity matrix format is tab separated
    corr_row_fields = corr_row.split('\t')
    
    #build a dictionary for every feature which is not in the blacklist
    for i in range(1, len(corr_row_fields) + 1):
            feat="feat" + str(i)
            if feat not in blacklist:
                D[feat] = dict()
    
    j = 1
    
    #fill the dictionary of dictionaries
    while (corr_row != ""):
        
        corr_row = corr_row.strip()
        corr_row_fields = corr_row.split('\t')
        
        for i in range(0, len(corr_row_fields)):
            
            feat1 = "feat" + str(j)
            feat2 = "feat" + str(i+1)
            
            if (feat1 not in blacklist) and (feat2 not in blacklist):
                if feat1 != feat2:
                    D[feat1][feat2] = abs(float(corr_row_fields[i]))
                    
        j+=1
        corr_row = corr_file.readline()

    #upload the relevance vector from the file
    r_file = open(relevance_path,'r')

    #build a dictionary containing relevance(feature(i),y)
    R = dict()

    #read the first informative line
    r_row = r_file.readline().strip()
    
    j = 1
    while (r_row != ""):
        
        feat = "feat" + str(j)
        r_row = r_row.strip()

        if feat not in blacklist:
            R[feat] = abs(float(r_row))
            
        j += 1
        r_row=r_file.readline()
    
    #initialize the feature subset list
    feature_list = []

    #start from the most y-correlated feature
    temp = max(R, key=R.get)
    
    #add to the feature subset list
    feature_list.append(int(temp.strip('feat')))
    
    while len(feature_list) < nr_feat:

        #spread penalities over the remaining features through the hyperparameter
        for d in D[temp]:
            R[d] = R[d] - 2*c*D[temp][d] 
        
        #remove selected from Relevance Vector
        R.pop(temp, 0)
        
        #remove selected from dictionary of dictionaries
        for d in D:
            D[d].pop(temp, 0)
        D.pop(temp, 0)
        
        #select the new feature to be added
        temp = max(R, key = R.get)
        
        feature_list.append(int(temp.strip('feat')))
            
    print feature_list
    lists.append(feature_list)
    
print lists

with open(root+'output/gas_selection.txt', "w") as outfile:
    for list_ele in lists:
        outfile.write(' '.join([str(ele) for ele in list_ele]) + "\n")
