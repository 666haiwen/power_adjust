import torch
import numpy as np

class Test(object):
    def __init__(self):
        self.a = 1
        self.b = [1,2,3]


testA = Test()
testB = Test()
testC = testA
print(testA.a, testA.b)
testC.a = 10
testC.b.append(10)
print(testC.a, testC.b)
print(testA.a, testA.b)
a = {
    'a': 1,
    'b': [1,2,3,4,5,6]
}
b = a.copy()
print(a)
b['a'] = 10
b['b'].append(7)
print(b)
print(a)
