#Use Logistic Regression to classify random numbers
import numpy as np
np.random.seed(12)
x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], 5000)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], 5000)
train_features = np.vstack((x1, x2))
train_labels = np.hstack((np.zeros(5000), np.ones(5000)))
def sigmoid(linear):
    return 1 / (1 + np.exp(-linear))
def logistic_regression(features, labels, num_steps, learning_rate):
    weights = np.random.rand(3)
    for step in range(num_steps):
        index = np.random.randint(0, 10000)
        features1 = np.hstack((train_features[index], 1))
        linear = np.dot(weights, features1)
        gradient = np.dot(train_labels[index] - sigmoid(linear), features1)
        weights += learning_rate * gradient
    return weights
x3 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], 5000)
x4 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], 5000)
test_features = np.vstack((x3, x4))  
test_labels = np.hstack((np.zeros(5000), np.ones(5000)))
weights = logistic_regression(test_features, test_labels, 50000, 5e-4)
final_linear = np.dot(np.hstack((test_features, np.ones((10000, 1)))), weights)
preds = np.round(sigmoid(final_linear))
print('Accuracy: {0}'.format((preds == test_labels).sum() / 10000))