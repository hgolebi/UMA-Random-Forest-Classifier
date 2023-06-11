# autorzy: Hubert Gołębiowski, Jakub Rozkosz

from dataset_preparation import Dataset
from testing_functions import run_all 


if __name__ == "__main__":
    # comparing results of predictions for classic and own implementation
    # - running for different numbers of trees
    mushrooms_dataset = Dataset("agaricus-lepiota.csv", 0.2)
    run_all(mushrooms_dataset, ('e', 'p'), 5)
    print("mushrooms done")
    heart_dataset = Dataset("heart.csv", 0.2)
    run_all(heart_dataset, ('0', '1'), 5)
    print("heart done")

    # checking if there has been an overfitting
    run_all(mushrooms_dataset, ('e', 'p'), 5, True)
    run_all(heart_dataset, ('0', '1'), 5, True)

    # checking an impact on results of different dataset division size (training/test set)
    dataset20to80 = Dataset("heart.csv", 0.2)
    run_all(dataset20to80, ('0', '1'), 5, True)
    dataset30to70 = Dataset("heart.csv", 0.3)
    run_all(dataset30to70, ('0', '1'), 5, True)
    dataset40to60 = Dataset("heart.csv", 0.4)
    run_all(dataset40to60, ('0', '1'), 5, True)
    dataset50to50 = Dataset("heart.csv", 0.5)
    run_all(dataset50to50, ('0', '1'), 5, True)