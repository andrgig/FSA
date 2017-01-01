#!/bin/bash

ROOT_DIR=$1
SELECTED_FEATURES_FILENAME=$2

python test_feature_selection_performance/test_performance_lambdamart.py $ROOT_DIR $SELECTED_FEATURES_FILENAME

# TODO: inserire lettura lista trials...

exit 0
