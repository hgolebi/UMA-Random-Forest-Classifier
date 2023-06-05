from random import randrange, shuffle

class Dataset:
    def __init__(self, filename):
        self.data = None
        self.attributes = None
        self.training_set = None
        self.training_size = None
        self.test_set = None
        self.test_size = None
        self.readFromFile(filename)
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

        self.test_set = self.data[:(self.size // 5)]
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



d = Dataset("agaricus-lepiota.data")
# d = Dataset("datatest.data")
pass

