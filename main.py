from classifier import Classifier, RandomForest
from dataset import Dataset
from sklearn.ensemble import RandomForestClassifier as ClassicRandomForest
from tabulate import tabulate, SEPARATING_LINE

FILE = "agaricus-lepiota.data"
class_set = ('e', 'p')
dataset = Dataset(FILE)

training_set, test_set = dataset.convertToNumbers()

n = 10
trainX = [row[1:] for row in training_set]
trainY = [row[0] for row in training_set]
testX = [row[1:] for row in test_set]
testY = [row[0] for row in test_set]


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

our = []
classic = []
for trees_count in [1, 5, 10, 15]:
    tp, tn, fp, fn = (0, 0, 0, 0)
    TP, TN, FP, FN = (0, 0, 0, 0)
    for n in range(5):
        TP_, TN_, FP_, FN_ = test_classic_implementation(trees_count, trainX, trainY, testX, testY, ord(class_set[0]))
        TP += TP_
        TN += TN_
        FP += FP_
        FN += FN_

        rf = RandomForest(dataset, trees_count)
        tp_, tn_, fp_, fn_ = rf.test()
        tp += tp_
        tn += tn_
        fp += fp_
        fn += fn_
    acc, prec = calculate_results((tp, tn, fp, fn))
    our.append([trees_count, tp, tn, fp, fn, acc, prec])

    ACC, PREC = calculate_results((TP, TN, FP, FN))
    classic.append([trees_count, TP, TN, FP, FN, ACC, PREC])

print('OUR IMPLEMENTATION')    
print(create_table(our))

print('CLASSIC IMPLEMENTATION')    
print(create_table(classic))

# res = test_classic_implementation(3, trainX, trainY, testX, testY, ord('e'))
