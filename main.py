# autorzy: Hubert Gołębiowski, Jakub Rozkosz

from classifier import Classifier, RandomForest
from dataset_preparation import Dataset
from sklearn.ensemble import RandomForestClassifier as ClassicRandomForest
from tabulate import tabulate, SEPARATING_LINE

def create_table(column_list):
    first_col = ['Trees', 'TP', 'TN', 'FP', 'FN', 'Acc%', 'Prec%']
    flist = zip(*([first_col] + column_list))
    table = tabulate(flist, tablefmt='simple_grid', numalign='right')
    return(table)
    

def test_classic_implementation(trees_num, train_x, train_y, test_x, test_y, positive_val):
    model = ClassicRandomForest(trees_num)
    model.fit(train_x, train_y)
    results = model.predict(test_x)

    tp, tn, fp, fn = (0, 0, 0, 0)

    for idx, result in enumerate(results):    
        if result == positive_val and test_y[idx] == positive_val:
            tp += 1
        if result != positive_val and test_y[idx] != positive_val:
            tn += 1
        if result == positive_val and test_y[idx] != positive_val:
            fp += 1
        if result != positive_val and test_y[idx] == positive_val:
            fn += 1

    return (tp, tn, fp, fn)

def calculate_results(result_table):
    tp, tn, fp, fn = result_table
    acc = (tp + tn) / (tp + tn + fp + fn) * 100
    prec = tp / (tp + fp) * 100
    acc = round(acc, 2)
    prec = round(prec, 2)
    return acc, prec

def run_tests(trainX, trainY, testX, testY, dataset, class_set, tests_on_training_set=False):
    our = []
    classic = []
    for trees_count in [1, 5, 10, 15]:
        tp, tn, fp, fn = (0, 0, 0, 0)
        TP, TN, FP, FN = (0, 0, 0, 0)
        for n in range(5):
            TP_, TN_, FP_, FN_ = test_classic_implementation(trees_count, trainX, trainY, testX, testY, class_set[0])
            TP += TP_
            TN += TN_
            FP += FP_
            FN += FN_

            rf = RandomForest(dataset, trees_count)
            tp_, tn_, fp_, fn_ = rf.test(tests_on_training_set)
            tp += tp_
            tn += tn_
            fp += fp_
            fn += fn_
        acc, prec = calculate_results((tp, tn, fp, fn))
        our.append([trees_count, tp, tn, fp, fn, acc, prec])

        ACC, PREC = calculate_results((TP, TN, FP, FN))
        classic.append([trees_count, TP, TN, FP, FN, ACC, PREC])

    return our, classic

def run_all(dataset_file, class_set, tests_on_training_set=False):
    mushrooms_dataset = Dataset(dataset_file)

    mush_training_set = mushrooms_dataset.convertToNumbers(mushrooms_dataset.training_set)
    mush_test_set = mushrooms_dataset.convertToNumbers(mushrooms_dataset.test_set)

    n = 10
    mush_trainX = [row[1:] for row in mush_training_set]
    mush_trainY = [row[0] for row in mush_training_set]
    mush_testX = [row[1:] for row in mush_test_set]
    mush_testY = [row[0] for row in mush_test_set]

    our, classic = run_tests(mush_trainX, mush_trainY, mush_testX, mush_testY, mushrooms_dataset, class_set, tests_on_training_set)
    test_dataset_type = 'test' if tests_on_training_set else 'training'
    print(f'OUR IMPLEMENTATION on {dataset_file} {test_dataset_type} dataset')    
    print(create_table(our))
    print(f'CLASSIC IMPLEMENTATION on {dataset_file} {test_dataset_type} dataset')    
    print(create_table(classic))


if __name__ == "__main__":
    run_all("agaricus-lepiota.csv", (1.0, 2.0))

    run_all("heart.csv", (0.0, 1.0))

    # checking if there has been an overfitting
    run_all("agaricus-lepiota.csv", (1.0, 2.0), True)
    run_all("heart.csv", (0.0, 1.0), True)

