import utils, vocabulary, math

class NaiveBayes:

    def __init__(self, fileName: str, isOriginal: bool, smoothing=0.01):
        self.smoothing = smoothing
        self.fileName = fileName
        self.vocabulary = vocabulary.originalVocabulary(fileName) if isOriginal else vocabulary.filteredVocabulary(fileName)
        self.vocabularySize = len(self.vocabulary)
        self.yesPrior = utils.total_yes_no(fileName) / (utils.total_yes_no(fileName) + utils.total_yes_no(fileName, False))
        self.noPrior = utils.total_yes_no(fileName, False) / (utils.total_yes_no(fileName) + utils.total_yes_no(fileName, False))
        self.numberOfYes = utils.total_word_in_class(self.vocabulary)
        self.numberOfNo = utils.total_word_in_class(self.vocabulary, False)

    def predict(self):
        '''
        Predicts the class of a tweet.
        '''
        data = utils.load_data(self.fileName)
        correct_prediction = 0
        trace_file_results = []

        for tweet in data:
            tokensList = tweet[1].split(" ")
            yes_score = math.log10(self.yesPrior) + sum([math.log10(self.conditional(token, 'yes')) for token in tokensList])
            no_score = math.log10(self.noPrior) + sum([math.log10(self.conditional(token, 'no')) for token in tokensList])
            
            best_score = yes_score if yes_score > no_score else no_score
            prediction = 'yes' if yes_score > no_score else 'no'
            true_value = tweet[2]
            label = 'correct' if prediction == true_value else 'wrong'

            if prediction == true_value:
                correct_prediction+=1

            print(f'{tweet[0]}  {prediction}  {best_score}  {true_value}  {label}')
            
        print(correct_prediction/len(data))

    def conditional(self, word: str, flag: str):
        '''
        Computes the conditional probability of a word given its class.
        '''
        wordOccurencesInClass = self.vocabulary[word][0] if flag == 'yes' else self.vocabulary[word][1]
        totalOccurencesInClass = self.numberOfYes if flag == 'yes' else self.numberOfNo
        return (wordOccurencesInClass + self.smoothing) / (totalOccurencesInClass + (self.vocabularySize * self.smoothing))


TRAINING_FILE = "data/covid_training.tsv"
TESTING_FILE = "data/covid_test_public.tsv"
model = NaiveBayes(TESTING_FILE, isOriginal=True)
model.predict()
