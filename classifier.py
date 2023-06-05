from reprlib import recursive_repr
from dataset import Dataset
import math

class Node:
    def __init__(self, value = None):
        self.value = value
        self.children = None

    def addChild(self, attribute_value, child_node):
        if self.children is None:
            self.children = dict()
        self.children[attribute_value] = child_node

class Classifier:
    def __init__(self, dataset):
        self.dataset = dataset
        attribute_index_list = list(range(1, len(dataset.attribute_list)))
        self.root = self.train(dataset.data, attribute_index_list)


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
        infGainlist = [-1 for n in self.dataset.attribute_list]
        entropy = self.entropy(data, self.dataset.attribute_list[0])
        for index in attribute_index_list:
            infGain = entropy
            for value in self.dataset.attribute_list[index]:
                new_data = []
                for elem in data:
                    if elem[index] == value:
                        new_data.append(elem)
                infGain -= len(new_data) / len(data) * self.entropy(new_data, self.dataset.attribute_list[0])
            infGainlist[index] = infGain
        return max(attribute_index_list, key = lambda x: infGainlist[x])


    def train(self, data, attribute_index_list):
        if len(data) == 0:
            return Node('?')

        if len(attribute_index_list) < 1:
            raise(Exception)

        only_class = data[0][0]
        for elem in data:
            if elem[0] != only_class:
                only_class = None
                break
        if not only_class is None:
            return Node(only_class)


        if len(attribute_index_list) == 1:
            class_count_list = []
            for clas in self.dataset.attribute_list[0]:
                count = 0
                for elem in data:
                    if elem[0] == clas:
                        count += 1
                class_count_list.append((clas, count))
            best_class = max(class_count_list, key=lambda x: x[1])
            return Node(best_class[0])

        attr_id = self.bestAttribute(data, attribute_index_list)
        attribute_index_list.remove(attr_id)
        node = Node(attr_id)
        for value in self.dataset.attribute_list[attr_id]:
            new_data = []
            for elem in data:
                if elem[attr_id] == value:
                    new_data.append(elem)
            node.addChild(value, self.train(new_data, attribute_index_list))
        return node

    def classify(self, object):
        curr_node = self.root
        while not curr_node.children is None:
            attr_index = curr_node.value
            attr_val = object[attr_index]
            curr_node = curr_node.children[attr_val]
        return curr_node.value

    def recursive_test(self, node):
        if node.children is None:
            return True

        attr_values = self.dataset.attribute_list[node.value]

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






