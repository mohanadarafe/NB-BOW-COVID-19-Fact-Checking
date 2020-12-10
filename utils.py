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
            dictionary[tokens] = [1,1,0] if isFactual else [1,0,1]
        else:
            dictionary[tokens][0] += 1
            if isFactual: dictionary[tokens][1] += 1
            else: dictionary[tokens][2] += 1
    return dictionary

def get_count(data, getFactual = True) -> int:
    '''
    Get the number of factual/non-factual claims.

    If getFactual is True, we return the count of yes's,
    otherwise, we return the count of no's.
    '''
    counter = 0
    value = 'yes' if getFactual else 'no'
    for tweet in data:
        if tweet[2] == value: counter += 1 
    return counter