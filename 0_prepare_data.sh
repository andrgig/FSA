#!/bin/bash

ROOT_DIR=$1/
NUM_FEATURES=$2

# generating intermediate data format
python prepare_data/prepare_data_from_svm.py $ROOT_DIR

# computing feature correlation from labels
python feature_relevance/compute_relevance_correlation-measures.py $ROOT_DIR $NUM_FEATURES

# computing feature relevance
python feature_relevance/compute_relevance_lambdamart.py $ROOT_DIR/output/sample_train_features.txt \
                                                         $ROOT_DIR/output/sample_validation_features.txt \
                                                         $ROOT_DIR/output/sample_test_features.txt \
                                                         $NUM_FEATURES \
                                                         $ROOT_DIR/output/feature_rank.txt \
                                                         $ROOT_DIR

# build blacklist
cat output/feature_rank.txt | grep nan | cut -f1 | cut -d't' -f2 | grep -v label > output/blacklist.txt
# TODO: add python script to produce blacklist

# computing pairwise feature similarity
python feature_similarity/compute_pairwise_feature_similarity_spearman.py $ROOT_DIR

exit 0
