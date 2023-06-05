from classifier import Classifier, RandomForest
from dataset import Dataset
from sklearn.ensemble import RandomForestClassifier as ClassicRandomForest

FILE = "agaricus-lepiota.data"
class_set = ('e', 'p')
dataset = Dataset(FILE)

for trees_count in range (10, 101, 10):
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
    print("TEST RESULTS FOR ", trees_count, ' TREES')
    print("True Positive: ", tp)
    print("True Negative: ", tn)
    print("False Positive: ", fp)
    print("False Negative: ", fn)
    print("Accuracy: ", acc, "%")
    print("Precision: ", prec, "%")


# n = 10
# our_implementation = RandomForest(dataset, n)
# classic_implementation = ClassicRandomForest(n)
# trainX = [row[1:] for row in dataset.training_set]
# trainY = [row[0] for row in dataset.training_set]
# classic_implementation.fit(trainX, trainY)
# testX = [row[1:] for row in dataset.test_set]
# testY = [row[0] for row in dataset.test_set]
# print(classic_implementation.score(testX, testY))

# def test_classic_implementation():
#     tp = 0
#     tn = 0
#     fp = 0
#     fn = 0

#     for elem in d.test_set:
#         clas = c.classify(elem)
#         real_class = elem[0]

#         if clas == class_set[0] and real_class == class_set[0]:
#             tp += 1

#         if clas == class_set[1] and real_class == class_set[1]:
#             tn += 1

#         if clas == class_set[0] and real_class == class_set[1]:
#             fp += 1

#         if clas == class_set[1] and real_class == class_set[0]:
#             fn += 1

#     acc = (tp + tn) / len(d.test_set) * 100

#     print("Tested data: ", DATA)
#     # print("Test: ", c.test())
#     print("True Positive: ", tp)
#     print("True Negative: ", tn)
#     print("False Positive: ", fp)
#     print("False Negative: ", fn)
#     print("Accuracy: ", acc, "%")