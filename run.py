import os
from naive_bayes import NaiveBayes

TRAINING_FILE = "data/covid_training.tsv"
TESTING_FILE = "data/covid_test_public.tsv"
def prompt_user():
    print("Welcome to the COVID-19 Fact Checking program")
    
    vocabularyType = input("Would you like to use the [o]riginal or [f]iltered vocabulary? (o/f): ")
    while(vocabularyType not in ["o", "f"]):
        print("Please enter a correct input.")
        vocabularyType = input("Would you like to use the [o]riginal or [f]iltered vocabulary? (o/f): ")

    return vocabularyType

if __name__ == "__main__":
    vocabularyType = prompt_user()
    useOriginal = True if vocabularyType == 'o' else False

    if not os.path.isdir("results"):
        os.makedirs("results")

    model = NaiveBayes(TRAINING_FILE, TESTING_FILE, originalVocabulary=useOriginal)
    model.predict()

    all_files = [file for file in os.listdir("results") if 'eval' in file]
    for file in all_files:
        fileDisplayTitle = 'Original Vocabulary Stats' if 'OV' in file else 'Filtered Vocabulary Stats'
        with open(f'results/{file}', 'r') as f:
            print(fileDisplayTitle)
            print(f'{f.read()}\n')

    print("Thank you for using our program.\nWritten by Mohanad Arafe & Ribal Aladeeb")