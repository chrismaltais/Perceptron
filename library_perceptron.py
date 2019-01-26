from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import csv

with open('data/testSeeds.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    features_train = []
    target_train = []
    for row in readCSV:
        row_floats = [float(feature) for feature in row[:-1]]
        features_train.append(row_floats) # First 7 features
        target_train.append(int(row[len(row) - 1])) # Labels

with open('data/trainSeeds.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    features_test = []
    target_test = []
    for row in readCSV:
        row_floats = [float(feature) for feature in row[:-1]]
        features_test.append(row_floats) # First 7 features
        target_test.append(int(row[len(row) - 1])) # Labels 

# Standard Scalar is used to standardize data
sc = StandardScaler()

# Train the scaler on the training data
sc.fit(features_train)

# Apply scaler to feature training data
features_train_std = sc.transform(features_train)

# Apply scaler to feature test data
features_test_std = sc.transform(features_test)

# Create perceptron object with 40 epochs and learning rate of 0.1
perceptron = Perceptron(max_iter=1000, eta0=0.1)

# Train perceptron
perceptron.fit(features_train_std,target_train)

# Run the trained perceptron on the test wheat data
target_prediction = perceptron.predict(features_test_std)

print('Accuracy: %.2f' % accuracy_score(target_test, target_prediction))
print('Train accuracy: %.2f' % perceptron.score(features_train_std, target_train))
print('Test accuracy: %.2f' % perceptron.score(features_test_std, target_test))




