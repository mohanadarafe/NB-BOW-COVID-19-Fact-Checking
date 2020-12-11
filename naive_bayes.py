import numpy as np
import utils, vocabulary, math

class NaiveBayes:

    def __init__(self, fileName: str, isOriginal: bool, smoothing=0.01):
        self.smoothing = smoothing
        self.fileName = fileName
        self.originalVocabulary = isOriginal
        self.vocabulary = vocabulary.originalVocabulary(fileName) if isOriginal else vocabulary.filteredVocabulary(fileName)
        self.vocabularySize = len(self.vocabulary)
        self.yesPrior = utils.total_yes_no(fileName) / (utils.total_yes_no(fileName) + utils.total_yes_no(fileName, False))
        self.noPrior = utils.total_yes_no(fileName, False) / (utils.total_yes_no(fileName) + utils.total_yes_no(fileName, False))
        self.numberOfYes = utils.total_word_in_class(self.vocabulary)
        self.numberOfNo = utils.total_word_in_class(self.vocabulary, False)
        self.trace_list = []

    def predict(self):
        '''
        Predicts the class of a tweet.
        '''
        conf_matrix = np.zeros(4).reshape(2,2)
        data = utils.load_data(self.fileName)
        correct_prediction = 0

        for tweet in data:
            tokensList = tweet[1].split()
            yes_score = math.log10(self.yesPrior) + sum([math.log10(self.conditional(token, 'yes')) for token in tokensList])
            no_score = math.log10(self.noPrior) + sum([math.log10(self.conditional(token, 'no')) for token in tokensList])
            
            best_score = yes_score if yes_score > no_score else no_score
            prediction = 'yes' if yes_score > no_score else 'no'
            true_value = tweet[2]
            label = 'correct' if prediction == true_value else 'wrong'

            if prediction == true_value:
                correct_prediction+=1
                conf_matrix[0][0] += 1
            elif prediction == 'yes' and true_value == 'no':
                conf_matrix[0][1] += 1
            elif prediction == 'no' and true_value == 'yes':
                conf_matrix[1][0] += 1
            else:
                conf_matrix[1][1] += 1

            self.trace_list.append(f'{tweet[0]}  {prediction}  {best_score}  {true_value}  {label}')

        self.eval_file(correct_prediction/len(data), conf_matrix)
        self.trace_file()

    def conditional(self, word: str, flag: str):
        '''
        Computes the conditional probability of a word given its class.
        '''
        if word not in self.vocabulary: wordOccurencesInClass = 0
        else: wordOccurencesInClass = self.vocabulary[word][0] if flag == 'yes' else self.vocabulary[word][1]
        totalOccurencesInClass = self.numberOfYes if flag == 'yes' else self.numberOfNo
        return (wordOccurencesInClass + self.smoothing) / (totalOccurencesInClass + (self.vocabularySize * self.smoothing))

    def eval_file(self, precision: float, conf_matrix):
        '''
        Produces the evaluation file for the model.
        '''
        assert len(self.trace_list) > 0, "Make sure you make predictions first!"
        vocabularyType = 'OV' if self.originalVocabulary else 'FV'
        TP = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]
        TN = conf_matrix[1][1]
        yes_P = TP / (TP + FP)
        no_P = TN / (TN + FN)
        yes_R = TP / (TP + FN)
        no_R = TN / (TN + FP)
        yes_F1 = TP / (TP + (0.5 * (FP + FN)))
        no_F1 = TN / (TN + (0.5 * (FP + FN)))
        with open(f"eval_NB_BOW_{vocabularyType}.txt", "w") as f:
            f.write(f'{round(precision, 4)}\n')
            f.write(f'{round(yes_P, 4)}  {round(no_P, 4)}\n')
            f.write(f'{round(yes_R, 4)}  {round(no_R, 4)}\n')
            f.write(f'{round(yes_F1, 4)}  {round(no_F1, 4)}')

    def trace_file(self):
        '''
        Produces the trace file for the model.
        '''
        assert len(self.trace_list) > 0, "Make sure you make predictions first!"
        vocabularyType = 'OV' if self.originalVocabulary else 'FV'

        with open(f"trace_NB_BOW_{vocabularyType}.txt", "w") as f:
            for res in self.trace_list:
                f.write(res + '\n')

TRAINING_FILE = "data/covid_training.tsv"
TESTING_FILE = "data/covid_test_public.tsv"
model = NaiveBayes(TESTING_FILE, isOriginal=True)
model.predict()