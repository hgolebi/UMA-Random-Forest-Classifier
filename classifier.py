from reprlib import recursive_repr
from dataset import Dataset
from collections import Counter
import math

class Node:
    def __init__(self, value = None):
        self.value = value
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

        # Sprawdzamy, czy w zbiorze została tylko 1 klasa
        only_class = data[0][0]
        for elem in data:
            if elem[0] != only_class:
                only_class = None
                break
        if not only_class is None:
            return Node(only_class)

        # Jeżeli nie ma już atrybutów, zwracamy najczęstszą klasę
        if len(attribute_index_list) == 1:
            class_column = [row[0] for row in data]
            counter = Counter(class_column)
            best_class = counter.most_common(1)[0][0]
            return Node(best_class[0])

        attr_id = self.bestAttribute(data, attribute_index_list)
        attribute_index_list.remove(attr_id)
        node = Node(attr_id)
        for attr_value in self.dataset.attributes[attr_id]:
            new_data = []
            for row in data:
                if row[attr_id] == attr_value:
                    new_data.append(row)
            node.addChild(attr_value, self.train(new_data, attribute_index_list))
        return node

    def entropy(self, list, class_set):
        entropy = 0
        for clas in class_set:
            class_count = 0
            for elem in list:
                if elem[0] == clas:
                    class_count += 1
            if class_count != 0:
                class_prob = class_count / len(list)
                entropy -= class_prob * math.log(class_prob)
        return entropy


    def bestAttribute(self, data, attribute_index_list):
        infGainlist = [-1 for n in self.dataset.attributes]
        entropy = self.entropy(data, self.dataset.attributes[0])
        for index in attribute_index_list:
            infGain = entropy
            for value in self.dataset.attributes[index]:
                new_data = []
                for elem in data:
                    if elem[index] == value:
                        new_data.append(elem)
                infGain -= len(new_data) / len(data) * self.entropy(new_data, self.dataset.attributes[0])
            infGainlist[index] = infGain
        return max(attribute_index_list, key = lambda x: infGainlist[x])


    def classify(self, object):
        curr_node = self.root
        while curr_node.children:
            attr_index = curr_node.value
            attr_val = object[attr_index]
            curr_node = curr_node.children[attr_val]
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






