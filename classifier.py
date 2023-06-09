# autorzy: Hubert Gołębiowski, Jakub Rozkosz

from reprlib import recursive_repr
from collections import Counter
import math
import random

class Node:
    def __init__(self, value = None):
        self.value = value  # Jezeli wezel ma dzieci, to value = indeks atrybutu decyzyjnego,
                            # jesli nie ma dzieci, to value = klasa    
        self.children = {}

    def addChild(self, attribute_value, child_node):
        self.children[attribute_value] = child_node
 

class Classifier:
    def __init__(self, dataset):
        self.dataset = dataset
        attribute_index_list = list(range(1, len(dataset.attributes)))
        self.root = self.train(dataset.data, attribute_index_list)


    def train(self, data, attribute_index_list):
        if len(data) == 0:
            return Node('?')

        if len(attribute_index_list) < 1:
            raise(Exception)

        # Pierwsza kolumna w zbiorze danych, to kolumna zawierająca klasy
        calculated_classcolumn = [row[0] for row in data]
        
        # Sprawdzamy, czy w zbiorze została tylko 1 klasa
        if all([calculated_class == calculated_classcolumn[0] for calculated_class in calculated_classcolumn]):
            return Node(calculated_classcolumn[0])

        # Jeżeli nie ma już atrybutów, zwracamy najczęstszą klasę
        if len(attribute_index_list) == 1:
            counter = Counter(calculated_classcolumn)
            best_class = counter.most_common(1)[0][0]
            return Node(best_class)

        # Wybieramy najlepszy atrybut, i tworzymy dla niego nowy węzeł decyzyjny
        attr_id = self.bestAttribute(data, attribute_index_list)
        attribute_index_list.remove(attr_id)
        node = Node(attr_id)
        for attr_value in self.dataset.attributes[attr_id]:
            new_data = [row for row in data if row[attr_id] == attr_value]
            node.addChild(attr_value, self.train(new_data, attribute_index_list))
        return node

    def entropy(self, list):
        entropy = 0
        calculated_classcolumn = [row[0] for row in list] 
        for calculated_class, count in Counter(calculated_classcolumn).items():
            calculated_classprobability = count / len(list)
            entropy -= calculated_classprobability * math.log(calculated_classprobability)
        return entropy


    def infGain(self, attr_index, data):
        inf_list = []
        for attr_value in self.dataset.attributes[attr_index]:
            new_data = [row for row in data if row[attr_index] == attr_value]
            inf_list.append(len(new_data) / len(data) * self.entropy(new_data))
        return self.entropy(data) - sum(inf_list)


    def bestAttribute(self, data, attribute_index_list):
        sample = random.sample(attribute_index_list, 2)
        return max(sample, key=lambda attr: self.infGain(attr, data))

    def classify(self, object):
        curr_node = self.root
        while curr_node.children:
            attr_index = curr_node.value
            attr_value = object[attr_index]
            curr_node = curr_node.children[attr_value]
        return curr_node.value

    def recursive_test(self, node):
        if not node.children:
            return True

        attr_values = self.dataset.attributes[node.value]

        if len(node.children) != len(attr_values):
            return False

        for child in node.children:
            if child not in attr_values:
                return False

        for child in node.children:
            is_good = self.recursive_test(node.children[child])
            if not is_good:
                return False
        return True

    def test(self):
        return self.recursive_test(self.root)


class RandomForest:
    def __init__(self, dataset, count):
        self.dataset = dataset
        self.count = count
        self.trees = []
        for i in range(count):
            self.trees.append(Classifier(dataset))
        
    def classify(self, object):
        decisions = [tree.classify(object) for tree in self.trees]
        counter = Counter(decisions)
        return counter.most_common(1)[0][0]


    def test(self):
        classes = [cls for cls in self.dataset.attributes[0]]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for row in self.dataset.test_set:
            calculated_class = self.classify(row)
            real_class = row[0]
            if calculated_class == classes[0] and real_class == classes[0]:
                tp += 1
            if calculated_class == classes[1] and real_class == classes[1]:
                tn += 1
            if calculated_class == classes[0] and real_class == classes[1]:
                fp += 1
            if calculated_class == classes[1] and real_class == classes[0]:
                fn += 1

        # acc = (tp + tn) / len(self.dataset.test_set) * 100
        # prec = tp / (tp + fp) * 100
        return (tp, tn, fp, fn)