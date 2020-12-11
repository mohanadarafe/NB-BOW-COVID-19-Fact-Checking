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

def build_vocabulary(dictionary: dict, tweet: list, sentiment: str) -> dict:
    '''
    The following function builds a vocabulary for every tweet & the
    sentiment associated with the tweet.
    '''
    isFactual = sentiment == 'yes'

    for tokens in tweet:
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
    Gets the total number of words in the factual class.
    '''
    counter = 0
    position = 0 if getYes else 1
    for key in vocabulary.keys():
        counter += vocabulary[key][position]

    return counter