from classifier_operations import GradBoostClassification
import numpy as np
from sklearn import datasets as ds

data_all = ds.load_breast_cancer()

x = data_all.data
y = data_all.target

y_names = data_all.target_names 

feature_names = data_all.feature_names
data_split = int(x.shape[0] * 0.6)

# high performance from few trees but doesn't improve with more
trees, step_size = 10, 0.1

threshold = 0.5

GB = GradBoostClassification(x, y, step_size, data_split)
for _ in range(trees):
    GB.current_tree_update()
actual = GB.y_train
predicted = GB.current_train_estimate
norm_pred = (predicted - np.min(predicted)) / (np.max(predicted) - np.min(predicted))
predicted_class = np.where(norm_pred > threshold, 1, 0)
correct_predictions = actual == predicted_class
percentage_correct = np.mean(correct_predictions) * 100

GB.test_all_trees(GB.trees, GB.X_test)
test_actual = GB.y_test
test_predicted = GB.current_test_estimate
norm_test_pred = (test_predicted - np.min(test_predicted)) / (np.max(test_predicted) - np.min(test_predicted))
test_predicted_class = np.where(norm_test_pred > threshold, 1, 0)
test_correct_predictions = test_actual == test_predicted_class
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