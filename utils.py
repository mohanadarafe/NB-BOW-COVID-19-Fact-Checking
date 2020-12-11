import os
import numpy as np
import pandas as pd
from nltk import word_tokenize

def load_data(filename: str):
    '''
    The following function loads the data set into a numpy array.
    '''
    assert os.path.isfile(filename), "The input file was not found!"
    header = None if 'test' in filename else 0
    df=pd.read_csv(filename, sep="\t", header=header)
    return df.values

def build_vocabulary(dictionary: dict, tokensList: list, sentiment: str) -> dict:
    '''
    The following function builds a vocabulary for every tweet & the
    sentiment associated with the tweet.
    '''
    isFactual = sentiment == 'yes'

    for tokens in tokensList:
        if tokens not in dictionary:
            dictionary[tokens] = [1,0] if isFactual else [0,1]
        else:
            if isFactual: dictionary[tokens][0] += 1
            else: dictionary[tokens][1] += 1
    return dictionary

def total_yes_no(fileName: str, getFactual = True) -> int:
    '''
    Get the number of factual/non-factual claims.

    If getFactual is True, we return the count of yes's,
    otherwise, we return the count of no's.
    '''
    data = load_data(fileName)
    counter = 0
    value = 'yes' if getFactual else 'no'
    for tweet in data:
        if tweet[2] == value: counter += 1 
    return counter

def total_word_in_class(vocabulary: dict, getYes = True) -> int:
    '''
    Gets the total number of words in a class.

    If getYes is true, we return the count of yes's,
    otherwise, we return the count of no's.
    '''
    counter = 0
    position = 0 if getYes else 1
    for key in vocabulary.keys():
        counter += vocabulary[key][position]

    return counter

def build_conf_matrix(prediction: str, true_value: str, conf_matrix):
    '''
    Actively builds the confusion matrix per tweet visited.
    '''
    if prediction == 'yes' and  true_value == 'yes':
        conf_matrix[0][0] += 1 #TP
    elif prediction == 'yes' and true_value == 'no':
        conf_matrix[0][1] += 1 #FP
    elif prediction == 'no' and true_value == 'yes':
        conf_matrix[1][0] += 1 #FN
    elif prediction == 'no' and true_value == 'no':
        conf_matrix[1][1] += 1 #TN

def get_metrics(conf_matrix) -> dict:
    '''
    Computes the metrics for evaluation
    '''
    TP = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TN = conf_matrix[1][1]
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    yes_P = TP / (TP + FP)
    no_P = TN / (TN + FN)
    yes_R = TP / (TP + FN)
    no_R = TN / (TN + FP)
    yes_F1 = TP / (TP + (0.5 * (FP + FN)))
    no_F1 = TN / (TN + (0.5 * (FP + FN)))

    metrics = {
        'accuracy': accuracy,
        'precision': {
            'yes': yes_P,
            'no': no_P
        },
        'recall': {
            'yes': yes_R,
            'no': no_R
        },
        'F1': {
            'yes': yes_F1,
            'no': no_F1
        }
    }
    return metrics