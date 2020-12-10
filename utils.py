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
    isYes = sentiment == 'yes'

    for tokens in tweet:
        if tokens not in dictionary:
            dictionary[tokens] = [1,1,0] if isYes else [1,0,1]
        else:
            dictionary[tokens][0] += 1
            if isYes: dictionary[tokens][1] += 1
            else: dictionary[tokens][2] += 1
    return dictionary
    