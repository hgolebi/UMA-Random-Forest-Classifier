from reprlib import recursive_repr
from dataset import Dataset
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
        class_column = [row[0] for row in data]
        
        # Sprawdzamy, czy w zbiorze została tylko 1 klasa
        if all([class_ == class_column[0] for class_ in class_column]):
            return Node(class_column[0])

        # Jeżeli nie ma już atrybutów, zwracamy najczęstszą klasę
        if len(attribute_index_list) == 1:
            counter = Counter(class_column)
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
        class_column = [row[0] for row in list] 
        for class_, count in Counter(class_column).items():
            class_probability = count / len(list)
            entropy -= class_probability * math.log(class_probability)
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






