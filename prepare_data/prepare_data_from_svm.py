#read raw train data from MSN dataset and produce a new file
#containing label    query id    feature1    feature2    ...    featureN
#(tab separated)

import sys

root = sys.argv[1]

label_array = []
query_array = []
feat_array = []
rpath = root + r'example_data/train.txt'

feat_file = open(rpath,'r')
counter = 0

feat_row = feat_file.readline()

while (feat_row != ""):
    
    feat_row = feat_row.strip()
    
    feat_row_fields = feat_row.split(':')
    
    for i in range(len(feat_row_fields)):

        feat_row_fields[i] = float(feat_row_fields[i].split(' ')[0])
    
    label_array.append(feat_row_fields[0])
    query_array.append(feat_row_fields[1])
    feat_array.append(feat_row_fields[2:len(feat_row_fields)])
    counter += 1
    feat_row = feat_file.readline()

print counter

#salvo un campione di dati su un file per esplorarne le caratteristiche

with open(root + r'output/sample_train_features.txt','w') as wfile:
    
    wfile.write("label\t")
    wfile.write("query\t")
    
    for element in range(1,len(feat_row_fields)-1):
        wfile.write("feat%s\t" % str(element))
    wfile.write("\n")
    
    for i in range(len(feat_array)):
        wfile.write("%s\t" % str(label_array[i]))
        wfile.write("%s\t" % str(query_array[i]))
        for element in range(len(feat_array[i])-1):
            wfile.write("%s\t" % str(feat_array[i][element]))
        wfile.write("%s\n" % str(feat_array[i][element+1]))
wfile.close()

label_array = []
query_array = []
feat_array = []

rpath = root + r'example_data/vali.txt'

feat_file = open(rpath,'r')
counter = 0

feat_row = feat_file.readline()

while (feat_row != ""):
    
    feat_row = feat_row.strip()
    
    feat_row_fields = feat_row.split(':')
    
    for i in range(len(feat_row_fields)):
        feat_row_fields[i] = float(feat_row_fields[i].split(' ')[0])
    
    label_array.append(feat_row_fields[0])
    query_array.append(feat_row_fields[1])
    feat_array.append(feat_row_fields[2:len(feat_row_fields)])
    
    counter += 1
    
    feat_row = feat_file.readline()

print counter


#salvo i dati su file 

with open(root + r'output/sample_validation_features.txt','w') as wfile:

    wfile.write("label\t")
    wfile.write("query\t")
    
    for element in range(1,len(feat_row_fields)-1):
        wfile.write("feat%s\t" % str(element))
    wfile.write("\n")
    
    for i in range(len(feat_array)):
        
        wfile.write("%s\t" % str(label_array[i]))
        wfile.write("%s\t" % str(query_array[i]))

        for element in range(len(feat_array[i])-1):
            wfile.write("%s\t" % str(feat_array[i][element]))
            
        wfile.write("%s\n" % str(feat_array[i][element+1]))
        
wfile.close()

label_array = []
query_array = []
feat_array = []
rpath = root + r'example_data/test.txt'

feat_file = open(rpath,'r')
counter = 0

feat_row=feat_file.readline()
while (feat_row != ""):

    feat_row = feat_row.strip()
    
    feat_row_fields = feat_row.split(':')
    
    for i in range(len(feat_row_fields)):

        feat_row_fields[i] = float(feat_row_fields[i].split(' ')[0])
    
    label_array.append(feat_row_fields[0])
    query_array.append(feat_row_fields[1])
    feat_array.append(feat_row_fields[2:len(feat_row_fields)])
    counter += 1
    feat_row = feat_file.readline()

print counter

#salvo i dati su file 

with open(root + r'output/sample_test_features.txt','w') as wfile:
    
    wfile.write("label\t")
    wfile.write("query\t")
    
    for element in range(1,len(feat_row_fields)-1):
        wfile.write("feat%s\t" % str(element))
    wfile.write("\n")
    
    for i in range(len(feat_array)):
        
        wfile.write("%s\t" % str(label_array[i]))
        wfile.write("%s\t" % str(query_array[i]))
        
        for element in range(len(feat_array[i])-1):
            wfile.write("%s\t" % str(feat_array[i][element]))
            
        wfile.write("%s\n" % str(feat_array[i][element+1]))
        
wfile.close()
