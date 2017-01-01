#!/bin/bash

ROOT_DIR=$1

# running GAS (Geng et al., ACM SIGIR 2007)
python feature_selection_competitors/GAS.py $ROOT_DIR

exit 0
