from random import randrange, shuffle

class Dataset:
    def __init__(self, filename):
        self.data = None
        self.attribute_list = None
        self.attribute_count = None
        self.training_set = None
        self.training_size = None
        self.test_set = None
        self.test_size = None
        self.readFromFile(filename)
        self.divideDataset()
        self.createAtributeSets()

    def readFromFile(self, filename):
        self.data = []
        f = open(filename, "r")
        text = f.read()
        if text == "":
            raise(Exception)
        rows = text.split('\n')
        if rows[-1] == '':
            rows.pop()
        shuffle(rows)
        for row in rows:
            self.data.append(row.split(","))
        self.size = len(self.data)
        self.attribute_count = len(self.data[0])

        self.test_set = self.data[:(self.size // 5)]
        self.training_set = self.data[(self.size // 5):]
        self.training_size = len(self.training_set)
        self.test_size = len(self.test_set)

    def createAtributeSets(self):
        self.attribute_list = []
        for i in range(self.attribute_count):
            newset = set()
            for elem in self.data:
                newset.add(elem[i])
            self.attribute_list.append(newset)



d = Dataset("agaricus-lepiota.data")
# d = Dataset("datatest.data")
pass

