import sys
import math
import numpy as np
import pandas as pd
from pprint import pprint

class Node(object):
    def __init__(self, attribute, threshold, left_nums=0, right_nums=0, leaf_nums=0):
        # Which column does it belong to
        self.attr = attribute
        # What is the threshold value that led to its split
        self.thres = threshold

        self.left = None
        self.right = None

        # Demographic data at that node
        self.left_nums = 0
        self.right_nums = 0
        self.leaf_nums = 0

        # Is it a leaf
        self.leaf = False
        # Label assigned to a node if leaf
        self.predict = None


# First select the threshold of the attribute to split set of test data on
# The threshold chosen splits the test data such that information gain is maximized
def select_threshold(df, attribute, predict_attr):
    # Convert dataframe column to a list and round each value
    values = df[attribute].tolist()
    values = [float(x) for x in values]
    # Remove duplicate values by converting the list to a set, then sort the set
    # values = set(values)
    # values = list(values)
    # values.sort()
    max_ig = float("-inf")
    thres_val = 0
    # try all threshold values that are half-way between successive values in this sorted list
    for i in range(0, len(values)):
        # thres = (values[i] + values[i+1])/2
        thres = values[i]
        ig = info_gain(df, attribute, predict_attr, thres)
        if ig >= max_ig:
            max_ig = ig
            thres_val = thres
    # Return the threshold value that maximizes information gained
    return thres_val

# Calculate info content (entropy) of the test data
def info_entropy(df, predict_attr):
    # Dataframe and number of positive/negatives examples in the data
    p_df = df[df[predict_attr] == 1]
    n_df = df[df[predict_attr] == 0]
    t = float(df.shape[0])
    p = float(p_df.shape[0])
    n = float(n_df.shape[0])
    proba = [p/t , n/t]

    # Calculate entropy
    if p == 0 or n == 0:
        I = 0
    else:
        I = -1 * np.dot(proba, np.log(proba))
        # I = ((-1*p)/(p + n))*math.log(p/(p+n), 2) + ((-1*n)/(p + n))*math.log(n/(p+n), 2)
    return I

# Calculates the weighted average of the entropy after an attribute test
def remainder(df, df_subsets, predict_attr):
    # number of test data
    num_data = df.shape[0]
    remainder = float(0)
    for df_sub in df_subsets:
        if df_sub.shape[0] > 1:
            remainder += float(df_sub.shape[0]/num_data)*info_entropy(df_sub, predict_attr)
    return remainder

# Calculates the information gain from the attribute test based on a given threshold
# Note: thresholds can change for the same attribute over time
def info_gain(df, attribute, predict_attr, threshold):
    sub_1 = df[df[attribute] < threshold]
    sub_2 = df[df[attribute] >= threshold]
    # Determine information content, and subract remainder of attributes from it
    ig = info_entropy(df, predict_attr) - remainder(df, [sub_1, sub_2], predict_attr)
    return ig

# Returns the number of positive and negative data
def num_class(df, predict_attr):
    p_df = df[df[predict_attr] == 1]
    n_df = df[df[predict_attr] == 0]
    return p_df.shape[0], n_df.shape[0]

# Chooses the attribute and its threshold with the highest info gain
# from the set of attributes
def choose_attr(df, attributes, predict_attr):
    max_info_gain = float("-inf")
    best_attr = None
    threshold = 0
    # Test each attribute (note attributes maybe be chosen more than once)
    for attr in attributes:
        thres = select_threshold(df, attr, predict_attr)
        ig = info_gain(df, attr, predict_attr, thres)
        if ig > max_info_gain:
            max_info_gain = ig
            best_attr = attr
            threshold = thres
    return best_attr, threshold

# Builds the Decision Tree based on training data, attributes to train on,
# and a prediction attribute
def build_tree(df, cols, predict_attr):

    # Get the number of positive and negative examples in the training data
    p, n = num_class(df, predict_attr)
    # Train data has one kind of value remaining
    if p == 0 or n == 0:
        # Create a leaf node indicating it's prediction
        leaf = Node(None,None, leaf_nums=(p+n))
        leaf.leaf = True
        if p >= n:
            leaf.predict = 1
        else:
            leaf.predict = 0
        return leaf
    else:
        # Otherwise, find the attribute for deciding with threshold
        best_attr, threshold = choose_attr(df, cols, predict_attr)
        # Create internal tree node based on attribute and it's threshold
        tree = Node(best_attr, threshold)

        sub_1 = df[df[best_attr] >= threshold]
        sub_2 = df[df[best_attr] < threshold]
        
        # Recursively build left and right subtree
        tree.left_nums = sub_1.shape[0]
        tree.right_nums = sub_2.shape[0]

        tree.left = build_tree(sub_1, cols, predict_attr)
        tree.right = build_tree(sub_2, cols, predict_attr)
        return tree

# Given a instance of a training data, make a prediction of healthy or colic
# based on the Decision Tree
# Assumes all data has been cleaned (i.e. no NULL data)
def predict(node, row_df):
    # If we are at a leaf node, return the prediction of the leaf node
    if node.leaf:
        return node.predict
    # Traverse left or right subtree based on instance's data
    if row_df[node.attr] <= node.thres:
        return predict(node.left, row_df)
    elif row_df[node.attr] > node.thres:
        return predict(node.right, row_df)

# Given a set of data, make a prediction for each instance using the Decision Tree
def test_predictions(root, df):
    num_data = df.shape[0]
    num_correct = 0
    for index,row in df.iterrows():
        prediction = predict(root, row)
        if prediction == row['Outcome']:
            num_correct += 1
    return round(num_correct/num_data, 2)





# Prints the tree level starting at given level
def print_tree(root, level):
    # print(counter*" ", end="")
    if root.leaf:
        pprint(root.predict, root.thres, level, root.leaf_nums)
    else:
        pprint(root.attr, root.thres, root.left_nums, root.right_nums, level)
    if root.left:
        print_tree(root.left, level + 1)
    if root.right:
        print_tree(root.right, level + 1)


def load_data(csv_file_name):
    
    df = pd.read_csv(csv_file_name, sep=" ", header=None)
    df.columns = ['x1', 'x2', 'Outcome']
    cols = df.columns
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df



def main():
    # An example use of 'build_tree' and 'predict'
    df_train = load_data('D1.txt')
    attributes = ['x1', 'x2']
    # attributes =  ['K', 'Na', 'CL', 'HCO', 'Endotoxin', 'Anioingap', 'PLA2', 'SDH', 'GLDH', 'TPP', 'Breath rate', 'PCV', 'Pulse rate', 'Fibrinogen', 'Dimer', 'FibPerDim']
    root = build_tree(df_train, attributes, 'Outcome')

    # print("Accuracy of test data")
    # df_test = load_data('horseTest.txt')
    # print(str(test_predictions(root, df_test)*100.0) + '%')

if __name__ == '__main__':
    main()
