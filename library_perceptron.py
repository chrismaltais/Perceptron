from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import os

def get_features_and_target_data(file):
    with open(file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        features = []
        targets = []
        for row in readCSV:
            row_floats = [float(feature) for feature in row[:-1]]
            features.append(row_floats) # First 7 features
            targets.append(int(row[len(row) - 1])) # Labels
        return features, targets

if __name__ == "__main__":
    features_train, target_train = get_features_and_target_data('data/trainSeeds.csv')
    features_test, target_test = get_features_and_target_data('data/testSeeds.csv')

    # Standard Scalar is used to standardize data
    sc = StandardScaler()

    # Train the scaler on the training data
    sc.fit(features_train)

    # Apply scaler to feature training data
    features_train_std = sc.transform(features_train)

    # Apply scaler to feature test data
    features_test_std = sc.transform(features_test)

    # Create perceptron object with 40 epochs and learning rate of 0.1
    perceptron = Perceptron(max_iter=1000, eta0=0.001)

    # Initialize training accuracy to be 0
    training_accuracy = 0
    
    # Must train perceptron until model has 96% accuracy on training data
    while(training_accuracy < 96):
        # Train perceptron
        perceptron.fit(features_train_std,target_train)

        # Run the trained perceptron on the test wheat data
        target_prediction = perceptron.predict(features_test_std)

        training_accuracy = perceptron.score(features_train_std, target_train) * 100
    
    # Test on testing data
    testing_accuracy = perceptron.score(features_test_std, target_test) * 100

    confusion_matrix_results = confusion_matrix(target_test, target_prediction)

    names = ['Kama', 'Rosa', 'Canadian']
    classification_report_results = classification_report(target_test, target_prediction, target_names=names)

    filename = 'results/libraryPerceptronResults.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w+') as f:
        f.write('Tool: SciKit Learn\n\n')
        f.write('Training Set Accuracy: %.2f \n' % training_accuracy)
        f.write('Testing Set Accuracy: %.2f \n\n' % testing_accuracy)
        f.write('Confusion Matrix: \n')
        f.write(np.array2string(confusion_matrix_results))
        f.write('\n\nClassification Report: \n')
        f.write(classification_report_results)




