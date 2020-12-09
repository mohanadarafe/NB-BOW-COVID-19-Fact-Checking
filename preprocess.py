import os
import numpy as np
import pandas as pd

def load_data(filename):
    '''
    The following function loads the data set into a numpy array.
    '''
    df=pd.read_csv(filename, sep="\t")
    return df.values

def preprocess_data(fileName: str):
    '''
    Preprocess the data to only keep the columns that interest us.
    '''
    assert os.path.isfile(fileName), "The file you inputted does not exist."
    data = load_data(fileName)
    X = data[:, 1]
    y = convert(data[:, 2])
    return X, y

def convert(data):
    '''
    The following function converts yes/no's to 1's/0's
    '''
    data = np.where(data == "yes", 1, data)
    data = np.where(data == "no", 0, data)
    return data.astype(int)

if __name__ == "__main__":
    X, y = preprocess_data("data/covid_training.tsv")
    print(X[0:5])
    print(y[0:5])