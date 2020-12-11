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
        if vocabulary[key][0] + vocabulary[key][1] <= 1:
            vocabulary.pop(key)

    return vocabulary

if __name__ == "__main__":
    TRAINING_FILE = "data/covid_training.tsv"
    TESTING_FILE = "data/covid_test_public.tsv"
    d1 = originalVocabulary(TRAINING_FILE)
    d2 = filteredVocabulary(TRAINING_FILE)
    print(f'Size of Original Vocabulary: {len(d1)}')
    print(f'Size of Filtered Vocabulary: {len(d2)}')
    print(f'Count of words in YES class in OV: {utils.total_word_in_class(d1)}')
    print(f'Count of words in NO class in OV: {utils.total_word_in_class(d1, False)}')
    print(f'Count of words in YES class in FV: {utils.total_word_in_class(d2)}')
    print(f'Count of words in NO class in FV: {utils.total_word_in_class(d2, False)}')
    print(f'Count of YES tweets: {utils.total_yes_no(TRAINING_FILE)}')
    print(f'Count of YES tweets: {utils.total_yes_no(TRAINING_FILE, False)}')