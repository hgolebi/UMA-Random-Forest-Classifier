from classifier import Classifier, RandomForest
from dataset import Dataset
from sklearn.ensemble import RandomForestClassifier as ClassicRandomForest
from tabulate import tabulate

FILE = "agaricus-lepiota.data"
class_set = ('e', 'p')
dataset = Dataset(FILE)

training_set, test_set = dataset.convertToNumbers()

n = 10
trainX = [row[1:] for row in training_set]
trainY = [row[0] for row in training_set]
testX = [row[1:] for row in test_set]
testY = [row[0] for row in test_set]

def create_table(method, column_list):
    first_col = ['Trees', 'TP', 'TN', 'FP', 'FN', 'Acc%', 'Prec%']
    table = tabulate([first_col] + column_list)
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

columns = []
for trees_count in [1, 5, 10, 15]:
    tp, tn, fp, fn = (0, 0, 0, 0)
    for n in range(5):
        rf = RandomForest(dataset, trees_count)
        tp_, tn_, fp_, fn_ = rf.test()
        tp += tp_
        tn += tn_
        fp += fp_
        fn += fn_
    acc = (tp + tn) / (tp + tn + fp + fn) * 100
    prec = tp / (tp + fp) * 100
    columns.append([trees_count, tp, tn, fp, fn, acc, prec])
print(create_table('Our', columns))

# res = test_classic_implementation(3, trainX, trainY, testX, testY, ord('e'))
