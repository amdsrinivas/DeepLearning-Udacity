# DeepLearning-Udacity
Udacity course on DeepLearning

Use udacity/ jupyter notebooks to  download the datasets.

model/model_building.py uses LogisticRegression of sklearn

model/neural-net-1layer.py uses neural netowrk one hidden layer of 1024 nodes ( using Tensorflow)

model/pkl files are the trianed models of sklearn. Use sklearn.externals.joblib to load them.
 Ex: model = joblib.load('pickelfile')