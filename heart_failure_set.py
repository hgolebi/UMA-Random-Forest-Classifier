# autorzy: Hubert Gołębiowski, Jakub Rozkosz

from random import randrange, shuffle
import csv

class HeartDataset:
    def __init__(self, filename, shift_done=True):
        self.data = []
        self.attributes = None
        self.training_set = None
        self.training_size = None
        self.test_set = None
        self.test_size = None
        if not shift_done:
            self.shift_class_column_to_first(filename)
        self.readFromFile(filename)
        self.unique_values = self.count_unique_values(filename)
        self.column_names = list(self.unique_values.keys())
        self.createAtributeSets()
    
    def shift_class_column_to_first(self, filename):
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)

            for row in rows:
                last_column = row.pop(-1)
                row.insert(0, last_column)

            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)

    def readFromFile(self, filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for line in reader:
                # shuffle(rows)
                self.data.append(line)
        self.size = len(self.data)

        self.test_set = self.data[1:(self.size // 5)]
        self.training_set = self.data[(self.size // 5):]
        self.training_size = len(self.training_set)
        self.test_size = len(self.test_set)

    def createAtributeSets(self):
        self.attributes = []
        for i in range((len(self.data[0]))):
            newset = set()
            for elem in self.data:
                newset.add(elem[i])
            self.attributes.append(newset)
    
    def count_unique_values(self, filename):
        unique_values = {}

        with open(filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)

            for column_name in header:
                unique_values[column_name] = set()
            for row in reader:
                for i, value in enumerate(row):
                    column_name = header[i]
                    unique_values[column_name].add(value)

        return unique_values
    
    def convertToNumbers(self, dataset):
        new_dataset = []
        for row in dataset:
            new_row = []
            for value, col_name in zip(row, self.column_names):
                try:
                    value = float(value)
                except ValueError:
                    for i, val in enumerate(self.unique_values[col_name]):
                        if val == value:
                            value = float(i+1)
                            break
                new_row.append(value)
            new_dataset.append(new_row)
        return new_dataset

d = HeartDataset("heart.csv", True)

pass

