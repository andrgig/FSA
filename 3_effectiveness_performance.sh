#!/bin/bash

ROOT_DIR=$1/
NUM_FEATURES=$2

python test_feature_selection_performance/test_performance_lambdamart.py $ROOT_DIR/sample_train_features.txt \
                                                                         $ROOT_DIR/sample_validation_features.txt \
                                                                         $ROOT_DIR/sample_test_features.txt \
                                                                         $ROOT_DIR

# TODO: inserire lettura lista trials...

exit 0
