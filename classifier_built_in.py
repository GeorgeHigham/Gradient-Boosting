import numpy as np
from sklearn import datasets as ds
from sklearn.ensemble import GradientBoostingClassifier

data_all = ds.load_breast_cancer()

x = data_all.data
y = data_all.target

data_split = int(x.shape[0] * 0.6)


X_train = x[:data_split,:]
X_test = x[data_split:,:]

y_train = y[:data_split]
y_test = y[data_split:]

model = GradientBoostingClassifier()

model.fit(X_train, y_train) 
y_train_pred = model.predict(X_train) 
y_test_pred = model.predict(X_test) 
correct_predictions = y_train_pred == y_train
test_correct_predictions = y_test_pred == y_test
percentage_correct = np.mean(correct_predictions) * 100
test_percentage_correct = np.mean(test_correct_predictions) * 100

print(f'Train Classification Accuracy: {percentage_correct:.2f}')
print(f'Test Classification Accuracy: {test_percentage_correct:.2f}')


"""
Self-built
Train Classification Accuracy: 93.55
Test Classification Accuracy: 92.54
"""
"""
Built In
Train Classification Accuracy: 100.00
Test Classification Accuracy: 95.18
"""