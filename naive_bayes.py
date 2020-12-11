import numpy as np
import utils, vocabulary, math

class NaiveBayes:

    def __init__(self, trainFilename: str, testFilename: str, originalVocabulary: bool, smoothing=0.01):
        self.smoothing = smoothing
        self.trainFilename = trainFilename
        self.testFilename = testFilename
        self.originalVocabulary = originalVocabulary
        self.vocabulary = vocabulary.originalVocabulary(trainFilename) if originalVocabulary else vocabulary.filteredVocabulary(trainFilename)
        self.vocabularySize = len(self.vocabulary)
        self.yesPrior = utils.total_yes_no(trainFilename) / (utils.total_yes_no(trainFilename) + utils.total_yes_no(trainFilename, False))
        self.noPrior = utils.total_yes_no(trainFilename, False) / (utils.total_yes_no(trainFilename) + utils.total_yes_no(trainFilename, False))
        self.numberOfYes = utils.total_word_in_class(self.vocabulary)
        self.numberOfNo = utils.total_word_in_class(self.vocabulary, False)
        self.trace_list = []
    
    def predict(self):
        '''
        Predicts the class of a tweet.
        '''
        conf_matrix = np.zeros(4).reshape(2,2)
        data = utils.load_data(self.testFilename)

        for tweet in data:
            tokensList = tweet[1].lower().split()
            yes_score = math.log10(self.yesPrior) + sum([math.log10(self.conditional(token, 'yes')) for token in tokensList if token in self.vocabulary])
            no_score = math.log10(self.noPrior) + sum([math.log10(self.conditional(token, 'no')) for token in tokensList if token in self.vocabulary])

            best_score = yes_score if yes_score > no_score else no_score
            prediction = 'yes' if yes_score > no_score else 'no'
            true_value = tweet[2]
            label = 'correct' if prediction == true_value else 'wrong'
            
            utils.build_conf_matrix(prediction, true_value, conf_matrix)
            self.trace_list.append(f'{tweet[0]}  {prediction}  {best_score}  {true_value}  {label}')

        self.eval_file(conf_matrix)
        self.trace_file()

    def conditional(self, word: str, flag: str):
        '''
        Computes the conditional probability of a word given its class.
        '''
        wordOccurencesInClass = self.vocabulary[word][0] if flag == 'yes' else self.vocabulary[word][1]
        totalOccurencesInClass = self.numberOfYes if flag == 'yes' else self.numberOfNo
        return (wordOccurencesInClass + self.smoothing) / (totalOccurencesInClass + (self.vocabularySize * self.smoothing))

    def eval_file(self, conf_matrix):
        '''
        Produces the evaluation file for the model.
        '''
        assert len(self.trace_list) > 0, "Make sure you make predictions first!"
        vocabularyType = 'OV' if self.originalVocabulary else 'FV'
        metrics = utils.get_metrics(conf_matrix)
        with open(f"results/eval_NB_BOW_{vocabularyType}.txt", "w") as f:
            f.write(f'{round(metrics["accuracy"], 4)}\n')
            f.write(f'{round(metrics["precision"]["yes"], 4)}  {round(metrics["precision"]["yes"], 4)}\n')
            f.write(f'{round(metrics["recall"]["yes"], 4)}  {round(metrics["recall"]["no"], 4)}\n')
            f.write(f'{round(metrics["F1"]["yes"], 4)}  {round(metrics["F1"]["no"], 4)}')

    def trace_file(self):
        '''
        Produces the trace file for the model.
        '''
        assert len(self.trace_list) > 0, "Make sure you make predictions first!"
        vocabularyType = 'OV' if self.originalVocabulary else 'FV'

        with open(f"results/trace_NB_BOW_{vocabularyType}.txt", "w") as f:
            for res in self.trace_list:
                f.write(res + '\n')
