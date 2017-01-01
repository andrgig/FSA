#!/bin/bash

ROOT_DIR=$1/

# running naive greedy feature selection
python feature_selection/NGAS.py $ROOT_DIR 

# running improved greedy feature selection
python feature_selection/XGAS.py $ROOT_DIR 

# clustering data...
Rscript feature_selection/hierarchical_clustering/hierarchical_clustering_wald.R $ROOT_DIR 

# running feature selection based on hierarchical clustering
python feature_selection/HCAS.py $ROOT_DIR 
