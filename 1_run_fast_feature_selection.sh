#!/bin/bash

ROOT_DIR=$1/

# running naive greedy feature selection
python feature_selection/NGAS.py $ROOT_DIR 

# running improved greedy feature selection
python feature_selection/XGAS.py $ROOT_DIR 

# TODO: rendere script R parametrico nell'input
# clustering data...
Rscript feature_selection/hierarchical_clustering/hierarchical_clustering_wald.R $ROOT_DIR 

# building feature file before running HCAS
cat $ROOT_DIR/output/feature_rank.txt | grep -v "nan" | cut -f1 | cut -d't' -f2 | grep -v "label" > $ROOT_DIR/output/nonescludere.txt

# running feature selection based on hierarchical clustering
python feature_selection/HCAS.py $ROOT_DIR 
