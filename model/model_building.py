import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import set_printoptions , inf

set_printoptions(threshold=inf)
data = open("../notMNIST.pickle","rb")

values = pickle.load(data)

'''print(type(values))
print(values.keys())
print(values['train_dataset'][0])
'''
train_dataset = values['train_dataset']

nsamples, nx, ny = train_dataset.shape

train_dataset = train_dataset.reshape((nsamples, nx*ny))
print(train_dataset[0])

#plt.imsave('test1.png', values[10].reshape(28,28), cmap=cm.gray)
#plt.imshow(values[0].reshape(28,28))

model = LogisticRegression(solver='saga')
print("Started training")
model.fit(train_dataset[0:10000], values['train_labels'][0:10000])
print("FInished training")
joblib.dump(model,'initial_model_saga_10000.pkl')