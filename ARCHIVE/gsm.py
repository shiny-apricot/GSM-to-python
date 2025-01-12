# %% [markdown]
# # GSM to Python Project

# %%
# determine the current project folder
import pathlib

# determine the current project folder
project_folder = pathlib.Path().parent.absolute()
project_folder

# %% [markdown]
# # Seeds

# %%
# set specific seeds to get the exact same results across different experiments
import numpy as np
import random

np.random.seed(44)
random.seed(44)

# %% [markdown]
# # User Input Parameters

# %%
INPUT_FILE_NAME = "GDS1962.csv"
GROUP_FILE_NAME = "cancer-DisGeNET.txt"
GROUP_COLUMN_NAME = "diseaseName"

NUMBER_OF_ITERATION = 5
MODEL_NAME = "RandomForest" # DecisionTree, RandomForest, SVM, KNN, MLP
LABEL_OF_POSITIVE_CLASS = "pos"
LABEL_OF_NEGATIVE_CLASS = "neg"
CLASS_MIN_BALANCE_RATIO = 0.5 # this is to handle imbalanced data by undersampling the majority class 
# (e.g. if the ratio is 0.5, the majority class has 1000 samples, and the minority class has 200 samples, then the majority class will be undersampled to 400 samples)
TRAIN_TEST_RATIO = 0.7

NORMALIZATION = "minmax" # minmax, zscore
FILTER_FEATURES_BY_TTEST_TO = 1000 # 0 to disable
FILTER_BEST_X_GROUPS = 10

# %% [markdown]
# # Input

# %%
# read the csv input from data folder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the csv input from data folder
data_folder = project_folder / "data"
data_file = data_folder / INPUT_FILE_NAME
data = pd.read_csv(data_file)

# %%
# print the first 5 rows of the data
data.head()

# %%
# the input is like::
# class	MIR4640	PAX8	EPHB3	SPATA17	PRR22	PRR22 (#1)	SLC39A5	NEXN	MFAP3	...	FAXDC2 (#1)	SLC48A1 (#2)	DTX3 (#3)	SIGIRR (#1)	SIDT2 (#1)	HIF1AN (#2)	MIR1306 (#2)	FAM86DP	ADGRA2 (#1)	EHBP1L1 (#3)
# 0	neg	4701.5	1616.3	107.5	13.3	56.2	291.1	205.2	219.3	42.6	...	3960.7	2253.8	554.5	606.9	2954.2	495.1	422.6	441.5	336.8	215.8
# 1	neg	4735.0	1527.2	270.8	12.8	51.2	209.5	265.3	109.2	77.7	...	3799.2	2528.8	626.8	868.3	2760.5	425.5	488.2	359.7	376.4	269.6

# %% [markdown]
# ## Missing Values

# %%
# get rid of missing values and measure how many rows are removed
print("Before dropping missing values: ", data.shape)
data = data.dropna()
print("After dropping missing values: ", data.shape)

# %% [markdown]
# ## Group File

# %%
# get group file from data / group_file / {file}
# its format is like::
# geneSymbol	diseaseName
# A1BG	Glioblastoma
# A1BG	Meningioma
# A1BG	Subependymal Giant Cell Astrocytoma

group_file = data_folder / "group_file" / GROUP_FILE_NAME
group = pd.read_csv(group_file, sep="\t")
group.tail()

# %% [markdown]
# ### Get Unique Group Names

# %%
groups = group.iloc[:, 1].unique()

print("Number of groups: ", len(groups))
print("Groups: ", groups)

# %% [markdown]
# # ✅✅ GSM ✅✅

# %% [markdown]
# ## Sample by the smallest class

# %%
# Measure the amount of samples for each class.
# Then, pick the class with the least amount of samples and reduce the other classes to the same amount.
# This is to avoid the problem of imbalanced data.

# get the number of samples for each class
print("Number of samples for each class:")
print(data["class"].value_counts())

# get the class with the least amount of samples
min_samples = data["class"].value_counts().min()
print("Minimum number of samples: ", min_samples)

# reduce the other classes to the amount of specified ratio
data = data.groupby("class").head(min_samples / CLASS_MIN_BALANCE_RATIO)
print("Number of samples for each class after reducing:")
print(data["class"].value_counts())

# %% [markdown]
# ## Normalization

# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# apply normalization
if NORMALIZATION == "minmax":
    scaler = MinMaxScaler()
elif NORMALIZATION == "zscore":
    scaler = StandardScaler()
else:
    raise Exception("Invalid normalization method")

# get the data without the class column
data_without_class = data.drop("class", axis=1)

# normalize the data
data_normalized = scaler.fit_transform(data_without_class)

# convert the normalized data to a dataframe
data_normalized = pd.DataFrame(data_normalized, columns=data_without_class.columns)

# add the class column to the normalized data at the beginning
data_normalized.insert(0, "class", data["class"])


# %%
data_normalized.head(2)

# %%
# change the class column to numerical
data_normalized["class"].replace({LABEL_OF_NEGATIVE_CLASS: 0, LABEL_OF_POSITIVE_CLASS: 1}, inplace=True)

# %% [markdown]
# ## Filter Features by TTest

# %%
# # apply anova test to the each feature of the train data
# from scipy import stats
# from tqdm import tqdm

# def apply_anova_levene_test(data, FILTER_FEATURES_BY_TTEST_TO):
#     # Apply ANOVA test to each feature
#     anova_results = {}
    
#     for column in tqdm(data.columns[1:]):  # Skip the 'class' column
#         f_val, p_val = stats.f_oneway(data[data['class'] == 'pos'][column], data[data['class'] == 'neg'][column])
#         anova_results[column] = p_val

#     # Sort features by p-value in ascending order (lower p-value means more significant)
#     sorted_features = sorted(anova_results, key=anova_results.get)

#     # Select the top FILTER_FEATURES_BY_TTEST_TO features
#     top_features = sorted_features[:FILTER_FEATURES_BY_TTEST_TO]

#     return top_features

# %%
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(score_func=chi2, k=FILTER_FEATURES_BY_TTEST_TO)  # Select top 2 features
X_new = selector.fit_transform(data_normalized, data_normalized["class"])

# Get the selected feature names
selected_feature_names = data_normalized.columns[selector.get_support()]

# Convert the selected features to a DataFrame
data_filtered = pd.DataFrame(data_normalized, columns=selected_feature_names)

# %%
# apply one-way anova
# significant_features = apply_anova_levene_test(train, FILTER_FEATURES_BY_TTEST_TO)

# %%
# print(f"Significant features: {significant_features}")


# %%
# filter the data by the features
# train_v2 = train[["class"] + significant_features]
# test_v2 = test[["class"] + significant_features]

# test_v2.shape, train_v2.shape


# %%
# train_v2.head(2)


# %%
# count how many common features does filtered_features have with significant_features (old vs new method)
# common_features = len(set(filtered_features) & set(significant_features))
# print(f"Number of common features: {common_features}")

# %%
# from line_profiler import LineProfiler
# lp = LineProfiler()
# # lp.run("apply_anova_levene_test(train, FILTER_FEATURES_BY_TTEST_TO)")
# lp_wrapper = lp(apply_anova_levene_test)
# lp_wrapper(train, FILTER_FEATURES_BY_TTEST_TO)

# %%


# %%


# %% [markdown]
# ## Split Data by Train and Test

# %%
# split the data into train and test
from sklearn.model_selection import train_test_split

def split_data(data):
    train, test = train_test_split(data, test_size=1 - TRAIN_TEST_RATIO, stratify=data["class"])
    print("Number of training samples: ", len(train))
    print("Number of test samples: ", len(test))
    return train, test

train_v2, test_v2 = split_data(data_filtered)

# %% [markdown]
# ## Grouping

# %%
# some columns are like "PRR22 (#1)" or "SLC48A1 (#2)" 
# but we need to remove the (#1) or (#2) part in order to make the column name valid while matching them with the group file
import re
def remove_parenthesis(s):
    return re.sub(r'\s*\(.*\)\s*', '', s)


# %%
def get_features_of_each_group(group):
    unique_groups = group.iloc[:,1].unique()
    filters_of_unique_groups = {}
    for group_name in unique_groups:
        # get the features that are related to the group
        features_of_group = group[group.iloc[:, 1] == group_name].iloc[:, 0].to_list()

        filters_of_unique_groups[group_name] = features_of_group
    return filters_of_unique_groups

# %%
featureList_of_groups = get_features_of_each_group(group)

# %%
featureList_of_groups

# %%
# print the first 6 element of groups of features
for feature, groups in list(featureList_of_groups.items())[:6]:
    print(feature, groups)

# %% [markdown]
# ##  Scoring

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# split the data into train and test
def split_data(data):
    # get the features
    features = data.columns[1:]
    # get the class labels
    labels = data["class"]
    # split the data into train and test
    train_x, test_x, train_y, test_y = train_test_split(data[features], labels, test_size=0.2, random_state=42)
    # return the train and test data
    return train_x, test_x, train_y, test_y

# normalize the data
def normalize_data(train_x, test_x, NORMALIZATION):
    # normalize the data
    if NORMALIZATION == "minmax":
        # create the scaler
        scaler = preprocessing.MinMaxScaler()
    elif NORMALIZATION == "zscore":
        # create the scaler
        scaler = preprocessing.StandardScaler()
    else:
        raise Exception("Invalid normalization method")
    # fit the scaler
    scaler.fit(train_x)
    # normalize the train and test data
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    # return the normalized train and test data
    return train_x, test_x

# train the model
def train_model(train_x, train_y, MODEL_NAME):
    # train the model
    if MODEL_NAME == "DecisionTree":
        # create the model
        model = DecisionTreeClassifier()
    elif MODEL_NAME == "RandomForest":
        # create the model
        model = RandomForestClassifier()
    elif MODEL_NAME == "SVM":
        # create the model
        model = svm.SVC()
    elif MODEL_NAME == "KNN":
        # create the model
        model = KNeighborsClassifier()
    elif MODEL_NAME == "MLP":
        # create the model
        model = MLPClassifier()
    else:
        raise Exception("Invalid model name")
    # train the model
    model.fit(train_x, train_y)
    # return the model
    return model


# %%
# calculate the metrics
def calculate_metrics(test_y, predicted_y):
    # calculate the metrics
    accuracy = accuracy_score(test_y, predicted_y)
    precision = precision_score(test_y, predicted_y)
    recall = recall_score(test_y, predicted_y)
    f1 = f1_score(test_y, predicted_y,
                #   pos_label=LABEL_OF_POSITIVE_CLASS
                  )
    # return the metrics
    return accuracy, precision, recall, f1

# %%

# combine all the steps
def scoring_module(data: pd.DataFrame, NUMBER_OF_ITERATION: int, MODEL_NAME: str, NORMALIZATION: str):
    # create a list to store the metrics
    metrics = []
    # repeat the process NUMBER_OF_ITERATION times
    for i in range(NUMBER_OF_ITERATION):
        # split the data into train and test
        train_x, test_x, train_y, test_y = split_data(data)
        # normalize the data
        train_x, test_x = normalize_data(train_x, test_x, NORMALIZATION)
        # train the model
        model = train_model(train_x, train_y, MODEL_NAME)
        # predict the test data
        predicted_y = model.predict(test_x)
        # calculate the metrics
        accuracy, precision, recall, f1 = calculate_metrics(test_y, predicted_y)
        # store the metrics
        metrics.append([accuracy, precision, recall, f1])
    # return the metrics
    return metrics


# %%

# calculate the average of the metrics
def calculate_group_average_metrics(metrics):
    # calculate the average of the metrics
    return np.mean(metrics, axis=0)

# %%
train_v2.head()

# %% [markdown]
# ## Assign Ranks For Groups and Features

# %%
import pandas as pd
import re

def filter_dataframe_by_features(df_columns, feature_list):
    """
    Efficiently filter a DataFrame to include only columns that match the given feature list,
    allowing for extra information in parentheses.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame
    feature_list (list): List of features to keep
    
    Returns:
    pandas.DataFrame: A new DataFrame containing only the specified features
    """
    # Create a single regex pattern for all features
    pattern = r'^(' + '|'.join(re.escape(feature) for feature in feature_list) + r')(\s*\([^)]*\))?$'
    regex = re.compile(pattern)
    
    # Use vectorized operations to filter columns
    mask = df_columns.str.match(regex)
    columns_to_keep = df_columns[mask]
    
    # Return a new DataFrame with only the specified columns
    return columns_to_keep

# Example usage:
# features = ['A1BG', 'NAT2', 'SERPINA3', 'AAVS1', 'ABCA1', 'ABL1', 'ABO']
# filtered_data = filter_dataframe_by_features(data, features)

# %%


# %% [markdown]
# ## Apply Scoring to each Group

# %%
import pandas as pd
from tqdm.notebook import tqdm


# train a new model using each group of features.
# Then, calculate the average of the metrics
def apply_scoring_to_each_group(data, featureList_of_groups):
    # train a new model using each group of features.
    # Then, calculate the average of the metrics
    # create a list to store the metrics
    group_performance_dict = {}
    counter = 0
    for group_name, features in tqdm(list(featureList_of_groups.items())):
        counter += 1
        # give the features as list
        # match the column even if it include extra paranthesis (ADVANCED CODE)
        features = filter_dataframe_by_features(data.columns, features)
        # if the length of the features is 0, then skip
        if len(features) == 0:
            group_performance_dict[group_name] = []
            continue
        # add class to the beginning of each feature
        features.insert(0, 'class')
        data_temp = data[features]
        metrics = []
        # repeat the process NUMBER_OF_ITERATION times
        for i in range(NUMBER_OF_ITERATION):
            # split the data into train and test
            train_x, test_x, train_y, test_y = split_data(data_temp)
            # normalize the data
            train_x, test_x = normalize_data(train_x, test_x, NORMALIZATION)
            # train the model
            model = train_model(train_x, train_y, MODEL_NAME)
            # predict the test data
            predicted_y = model.predict(test_x)
            # calculate the metrics
            accuracy, precision, recall, f1 = calculate_metrics(test_y, predicted_y)
            # store the metrics
            metrics.append([accuracy, precision, recall, f1])
        # get the average of the metrics
        metrics = calculate_group_average_metrics(metrics)
        if counter > 4:
            break

    # return the metrics
    return metrics


# %%
all_group_feature_match_with_metrics = all_group_feature_match_with_metrics_original.reset_index(drop=True)
all_group_feature_match_with_metrics

# %% [markdown]
# ## Save Group Feature Matches with Metrics

# %%
# save all_group_feature_matches to a csv file in the output folder
output_folder = project_folder / "output"
output_folder.mkdir(parents=True, exist_ok=True)
output_file = output_folder / "all_group_feature_matches.csv"
all_group_feature_match_with_metrics.to_csv(output_file, index=False)

# %%
# sort the average of the metrics by accuracy
average_metrics_of_groups = average_metrics_of_groups.sort_values(by="accuracy", ascending=False)

# %%
average_metrics_of_groups

# %% [markdown]
# ### Save Ranked Groups

# %%
# save the average of the metrics to a file in the output folder
output_folder = project_folder / "output"
output_folder.mkdir(parents=True, exist_ok=True)
output_file = output_folder / "average_metrics_of_groups.txt"
average_metrics_of_groups.to_csv(output_file, sep="\t")


# %% [markdown]
# ## Remove Lower Scored Groups 

# %%
# Take the best performed groups
# Find the associated features to these groups and filter them
def filter_features_by_groups(data, groups_of_features, average_metrics_of_groups, FILTER_BEST_X_GROUPS):
    # get the best performed groups
    best_performed_groups = average_metrics_of_groups.index[:FILTER_BEST_X_GROUPS].to_list()
    # for each group, get associated features and store them in a list
    features_of_best_performed_groups = []
    for group in best_performed_groups:
        features_of_best_performed_groups += [feature for feature, groups in groups_of_features.items() if group in groups]
    # remove the duplicate features
    print("Number of features before removing duplicates: ", len(features_of_best_performed_groups))
    features_of_best_performed_groups = list(set(features_of_best_performed_groups))
    print("Duplicate features are removed. Number of features: ", len(features_of_best_performed_groups))
    # filter the data by the features
    data = data[["class"] + features_of_best_performed_groups]
    # return the filtered data and features_of_best_performed_groups
    return data, features_of_best_performed_groups

# %%
# filter the data by the features_of_best_performed_groups
train_v3, features_of_best_performed_groups  = filter_features_by_groups(train_v2, featureList_of_groups, average_metrics_of_groups, FILTER_BEST_X_GROUPS)
test_v3 = test_v2[["class"] + features_of_best_performed_groups]

# %%
features_of_best_performed_groups

# %%
train_v3.head(10)

# %% [markdown]
# ## Modeling

# %%
def calculate_metrics(test_y, predicted_y):
    # calculate the metrics
    accuracy = accuracy_score(test_y, predicted_y)
    precision = precision_score(test_y, predicted_y)
    recall = recall_score(test_y, predicted_y)
    f1 = f1_score(test_y, predicted_y)
    # return the metrics
    return accuracy, precision, recall, f1

# take the final data, train a model with it and calculate the metrics using test data
def final_scoring_module(train, test, MODEL_NAME):
    # split the data into labels and features
    train_x = train.drop("class", axis=1)
    train_y = train["class"]
    test_x = test.drop("class", axis=1)
    test_y = test["class"]
    # train the model
    model = train_model(train_x, train_y, MODEL_NAME)
    try:
        # predict the probability of test data
        predicted_y_proba = model.predict_proba(test_x)
        print(predicted_y_proba)
        # get the probability of the positive class
        predicted_y_proba = predicted_y_proba[:, 1]
    except:
        print("The probability of test data cannot be predicted. The model does not have the attribute 'predict_proba'")
        # predict the test data
        predicted_y_proba = model.predict(test_x)
    # convert the predicted_y_proba to binary
    predicted_y = np.where(predicted_y_proba > 0.5, 1, 0)
    # calculate the metrics
    accuracy, precision, recall, f1 = calculate_metrics(test_y, predicted_y)
    # return the metrics
    return accuracy, precision, recall, f1, predicted_y_proba

# %%
# calculate the metrics for the final data
accuracy, precision, recall, f1, predicted_y_proba = final_scoring_module(train_v3, test_v3, MODEL_NAME)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)
