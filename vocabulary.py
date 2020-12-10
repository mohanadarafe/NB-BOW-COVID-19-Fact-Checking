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
    vocabulary = originalVocabulary(filename)

    for key in list(vocabulary):
        if vocabulary[key][0] == 1:
            vocabulary.pop(key)

    return vocabulary

if __name__ == "__main__":
    d1 = originalVocabulary("data/covid_training.tsv")
    d2 = filteredVocabulary("data/covid_training.tsv")
    print(f'Size of Original Vocabulary: {len(d1)}')
    print(f'Size of Filtered Vocabulary: {len(d2)}')