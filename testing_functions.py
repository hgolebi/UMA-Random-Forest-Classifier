# autorzy: Hubert Gołębiowski, Jakub Rozkosz

from classifier import Classifier, RandomForest
from sklearn.ensemble import RandomForestClassifier as ClassicRandomForest
from tabulate import tabulate, SEPARATING_LINE

def create_table(column_list, division_size):
    print(f"{1-division_size} training set, {division_size} test set")
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

def run_tests_on_own(dataset, num_of_iters, tests_on_training_set=False):
    results = []
    for trees_count in [1, 5, 10, 25, 50, 100, 150]:
        tp, tn, fp, fn = (0, 0, 0, 0)
        for n in range(num_of_iters):
            rf = RandomForest(dataset, trees_count)
            tp_, tn_, fp_, fn_ = rf.test(tests_on_training_set)
            tp += tp_
            tn += tn_
            fp += fp_
            fn += fn_
        acc, prec = calculate_results((tp, tn, fp, fn))
        results.append([trees_count, tp, tn, fp, fn, acc, prec])

    return results


def run_tests_on_classic(trainX, trainY, testX, testY, class_set, num_of_iters):
    results = []
    for trees_count in [1, 5, 10, 25, 50, 100, 150]:
        tp, tn, fp, fn = (0, 0, 0, 0)
        for n in range(num_of_iters):
            tp_, tn_, fp_, fn_ = test_classic_implementation(trees_count, trainX, trainY, testX, testY, class_set[0])
            tp += tp_
            tn += tn_
            fp += fp_
            fn += fn_
        acc, prec = calculate_results((tp, tn, fp, fn))
        results.append([trees_count, tp, tn, fp, fn, acc, prec])

    return results

def run_all(dataset, class_set, num_of_iters, tests_on_training_set=False):
    training_set = dataset.convertToNumbers(dataset.training_set)
    test_set = dataset.convertToNumbers(dataset.test_set)

    trainX = [row[1:] for row in training_set]
    trainY = [row[0] for row in training_set]
    testX = [row[1:] for row in test_set]
    testY = [row[0] for row in test_set]

    test_dataset_type = 'test' if not tests_on_training_set else 'training'
    classic = run_tests_on_classic(trainX, trainY, testX, testY, class_set, num_of_iters)
    print(f'CLASSIC IMPLEMENTATION on {dataset.filename} {test_dataset_type} dataset')    
    print(create_table(classic, dataset.division_size))
    if tests_on_training_set:
        return 
    our = run_tests_on_own(dataset, num_of_iters, tests_on_training_set)
    print(f'OUR IMPLEMENTATION on {dataset.filename} {test_dataset_type} dataset')
    print(create_table(our, dataset.division_size))