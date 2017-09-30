import pickle
from random import sample
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from numpy import set_printoptions , inf


def accuarcy(a,b):
    i = 0
    count = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            count = count + 1
    return count/len(a)
set_printoptions(threshold=inf)
data = open("../notMNIST.pickle","rb")

values = pickle.load(data)

test_dataset = values['test_dataset']

nsamples , nx , ny = test_dataset.shape

test_dataset = test_dataset.reshape((nsamples,nx*ny))

#valid_dataset = sample(valid_dataset,10)
model = joblib.load('initial_model_saga_10000.pkl')

result = model.predict(test_dataset)

print(accuarcy(result,values['test_labels']))
#print(result)
#print(values['valid_labels'])
#print(type(model))

