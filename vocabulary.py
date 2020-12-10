import utils

def originalVocabulary(filename: str) -> dict:
    '''
    The following function will create the Original Vocabulary.

    We keep all the information we need using the following format:
    term -> [frequency, # of yes's, # of no's]
    '''
    data = utils.load_data(filename)
    vocabulary = dict()

    for tokens in data:
        tokensList = tokens[1].split(" ")
        sentiment = tokens[2]
        utils.build_vocabulary(vocabulary, tokensList, sentiment)

    return vocabulary

def filteredVocabulary(filename: str) -> dict:
    '''
    The following function will create the Filtered Vocabulary.

    We keep all the information we need using the following format:
    term -> [frequency, # of yes's, # of no's]
    '''
    data = utils.load_data(filename)
    vocabulary = dict()

    for tokens in data:
        tokensList = tokens[1].split(" ")
        sentiment = tokens[2]
        utils.build_vocabulary(vocabulary, tokensList, sentiment)

    for key in list(vocabulary):
        if vocabulary[key][0] == 1:
            vocabulary.pop(key)

    return vocabulary

if __name__ == "__main__":
    d1 = originalVocabulary()
    d2 = filteredVocabulary()