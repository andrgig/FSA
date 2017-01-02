# Fast Feature Selection for Learning to Rank

This is the code used in "Fast feature selection for Learning to Rank" (2016) by A. Gigli, C. Lucchese, R. Perego, F. Nardini  http://dl.acm.org/citation.cfm?id=2970433.

You can clone it ad run in bash through the following command

- `0_prepare_data.sh <YOU_DIRECTORY> <FEATURES_NUMBER>`

- `1_run_fast_feature_selection.sh <YOU_DIRECTORY>`

- `2_run_competitors.sh <YOU_DIRECTORY>`

- `3_effectiveness_performance.sh <YOU_DIRECTORY> <FEATURES_FILE>` 

where 

- `<YOU_DIRECTORY>` is the path of the directory where you have cloned this repository.
- `<FEATURES_NUMBER>` is the number of feature of the dataset (in this case 220).
- `<FEATURES_FILE>` is the filename of the file containing the selected features. In the paper we compare different Feature Selection Algorithms (FSAs): GAS, NGAS, XGAS and HCAS. In order to make the performance measurement more flexible `1_run_fast_feature_selection.sh <YOU_DIRECTORY>` generates 4 files: gas_selection_test.txt, ngas_selection_test.txt, xgas_selection_test.txt, hcas_selection_test.txt in the folder `/output`. Each file represents the selection of from a FSA and contains 7 feature selections, one for each features selection subset dimension (ie 5%, 10%, 20%, 30%, 40%, 50%, 75%).

The code demo is ready to be tested on sample data extracted from [*istella LETOR dataset](http://blog.istella.it/istella-learning-to-rank-dataset/). The dataset contains 220 features in SVM format.

**HINT**: If you simply want to run LambdaMART on the whole features set you can create a one-row file containing a sequence of integers from 1 to `<FEATURES_NUMBER>` separated by white spaces and then run

`python <DATA_DIRECTORY> <FEATURES_FILE>` 

where

- `<DATA_DIRECTORY>` is the directory where you have saved train, validation and test data in the appropriate format
- `<FEATURES_FILE>` is the one-row file containg the sequence of integers
