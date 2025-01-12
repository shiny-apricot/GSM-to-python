
# These are the input file names and the column name for the group data
INPUT_FILE_NAME = "GDS1962.csv"
GROUP_FILE_NAME = "cancer-DisGeNET.txt"
GROUP_COLUMN_NAME = "diseaseName"

# The number of iterations to run the model
NUMBER_OF_ITERATION = 5

# The model name to use for the model
# Options are DecisionTree, RandomForest, SVM, KNN, MLP
MODEL_NAME = "RandomForest"

# The labels for the positive and negative classes
LABEL_OF_POSITIVE_CLASS = "pos"
LABEL_OF_NEGATIVE_CLASS = "neg"

# The minimum balance ratio to use for undersampling the majority class
# (e.g. if the ratio is 0.5, the majority class has 1000 samples, and the minority class has 200 samples, 
# then the majority class will be undersampled to 400 samples)
CLASS_MIN_BALANCE_RATIO = 0.5

# The ratio to use for splitting the data into train and test sets
TRAIN_TEST_SPLIT_RATIO = 0.7

# The normalization method to use
# Options are minmax, zscore
NORMALIZATION_METHOD = "minmax"

# The number of features to select using t-test
FILTER_FEATURES_BY = 1000 # 0 to disable

# The number of best features to select from each group
FILTER_BEST_X_GROUPS = 10

# The random seed to use
RANDOM_SEED = 44

# The number of workers to use for parallel processing
NUM_WORKERS = 1
