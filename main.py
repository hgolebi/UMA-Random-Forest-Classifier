from classifier import Classifier
from dataset import Dataset


data1 = "agaricus-lepiota.data"
class1 = ('e', 'p')

data2 = "breast-cancer.data"
class2 = ("recurrence-events", "no-recurrence-events")

DATA = data2
CLASS = class2

d = Dataset(DATA)
c = Classifier(d)

tp = 0
tn = 0
fp = 0
fn = 0



for elem in d.test_set:
    clas = c.classify(elem)
    real_class = elem[0]

    if clas == CLASS[0] and real_class == CLASS[0]:
        tp += 1

    if clas == CLASS[1] and real_class == CLASS[1]:
        tn += 1

    if clas == CLASS[0] and real_class == CLASS[1]:
        fp += 1

    if clas == CLASS[1] and real_class == CLASS[0]:
        fn += 1

acc = (tp + tn) / len(d.test_set) * 100

print("Tested data: ", DATA)
print("Test: ", c.test())
print("True Positive: ", tp)
print("True Negative: ", tn)
print("False Positive: ", fp)
print("False Negative: ", fn)
print("Accuracy: ", acc, "%")