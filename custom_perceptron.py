from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import csv
import random
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

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i+1] * row[i]
    if activation >= 0:
        return 1
    else: 
        return 0

def fit_model(feature_data, target_data, weights, learning_rate, epochs):
    for iteration in range(epochs):
        for i in range(len(feature_data)):
            activation_neuron_one = predict(feature_data[i], weights[0])
            activation_neuron_two = predict(feature_data[i], weights[1])
            activation_neuron_three = predict(feature_data[i], weights[2])
            desired = [activation_neuron_one, activation_neuron_two, activation_neuron_three]
            error = target_data[i] - desired
            for j in range(len(error)):
                if error[j] != 0:
                    weights[j] = train_weights(weights[j], learning_rate, feature_data[i], error[j])
    return weights
     

def train_weights(weights, learning_rate, feature_data, error):
    weights[0] = weights[0] + (learning_rate * error * 1) # Adjust the bias
    for i in range(len(feature_data)):
        weights[i + 1] = weights[i + 1] + (learning_rate * error * feature_data[i])
    return weights

def create_random_weights(feature_data):
    row_length = len(feature_data) + 1
    weights = np.ones( (3,row_length) )
    for i in range(len(feature_data) + 1):
        weights[0][i] = random.uniform(-1, 1)
        weights[1][i] = random.uniform(-1, 1)
        weights[2][i] = random.uniform(-1, 1)
    return weights

def encode_target_values(target_data):
    encoded_targets = np.ones( (len(target_data), 3) )
    for i in range(len(target_data)):
        if target_data[i] == 1:
            encoded_targets[i] = [1, 0, 0]
        elif target_data[i] == 2:
            encoded_targets[i] = [0, 1, 0]
        elif target_data[i] == 3:
            encoded_targets[i] = [0, 0, 1]
    return encoded_targets

def decode_target_values(encoded_data):
    decoded_targets = []
    for i in range(len(encoded_data)):
        if np.array_equal(encoded_data[i], [1, 0, 0]):
            decoded_targets.append(1)
        elif np.array_equal(encoded_data[i], [0, 1, 0]):
            decoded_targets.append(2)
        elif np.array_equal(encoded_data[i], [0, 0, 1]):
            decoded_targets.append(3)
        else:
            decoded_targets.append(0)
    return decoded_targets

def predict_model(feature_data, weights):
    results = np.ones( (len(feature_data), 3) )
    for i in range(len(feature_data)):
        neuron_one = predict(feature_data[i], weights[0])
        neuron_two = predict(feature_data[i], weights[1])
        neuron_three = predict(feature_data[i], weights[2])
        results[i] = [neuron_one, neuron_two, neuron_three]
    return decode_target_values(results)

def score(predicted_values, target_values):
    success = 0
    fail = 0 
    for i in range(len(predicted_values)):
        if predicted_values[i] == target_values[i]:
            success = success + 1
        else:
            fail = fail + 1
    return success / (success + fail)

if __name__ == "__main__":
    features_train, target_train = get_features_and_target_data('data/trainSeeds.csv')
    features_test, target_test = get_features_and_target_data('data/testSeeds.csv')

    # Standard Scalar is used to standardize data
    sc = StandardScaler()

    # Train the scaler on the training data
    sc.fit(features_train)

    # Apply scaler to feature training data
    features_train_std = sc.transform(features_train)

    # Apply scaler to feature testing data
    features_test_std = sc.transform(features_test)
    
    # Initialize learning rate
    learning_rate = 0.001

    # Initialize number of epochs
    epochs = 1000
    num_epochs_elapsed = 0

    # Initialize training accuracy to 0
    training_accuracy = 0

    # Initializing weights
    initial_weights = create_random_weights(features_train_std[0])

    # Encode target data
    encoded_target_train = encode_target_values(target_train)

    # Preserve initial weights
    trained_weights = np.array(initial_weights)

    # Must train perceptron until model has 96% accuracy on training data
    print('Training until 96% training accuracy...')
    while (training_accuracy < 0.50):

        trained_weights = fit_model(features_train_std, encoded_target_train, trained_weights, learning_rate, epochs)

        predicted_training_targets = predict_model(features_train_std, trained_weights)
        
        training_accuracy = score(predicted_training_targets, target_train)

        print('Accuracy: %d' % (training_accuracy * 100))

        num_epochs_elapsed = num_epochs_elapsed + epochs

        print('Epochs: ', num_epochs_elapsed)

    # Use model with testing data
    print('Testing model on test data...')
    predicted_testing_targets = predict_model(features_test_std, trained_weights)
    testing_accuracy = score(predicted_testing_targets, target_test)
    print('Testing Accuracy: %d' % (testing_accuracy * 100))

    # Confusion Matrix
    confusion_matrix_results = confusion_matrix(target_test, predicted_testing_targets)

    # Precision and Recall
    names = ['Inavlid', 'Kama', 'Rosa', 'Canadian']
    classification_report_results = classification_report(target_test, predicted_testing_targets, target_names=names)

    # Convert results to strings to write to file...
    target_train_string = " ".join(str(x) for x in target_train)
    predicted_train_target_string = " ".join(str(x) for x in predicted_training_targets)
    target_test_string = " ".join(str(x) for x in target_test)
    predicted_test_target_string = " ".join(str(x) for x in predicted_testing_targets)

    filename = 'results/customPerceptronResults.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w+') as f:
        f.write('Custom Perceptron Results:\n\n')
        f.write('Training Set Accuracy: %.2f \n' % (training_accuracy * 100))
        f.write('Testing Set Accuracy: %.2f \n\n' % (testing_accuracy * 100))
        f.write('Initial weights:\n')
        f.write('Neuron One:\n')
        f.write(np.array2string(initial_weights[0]))
        f.write('\n')
        f.write('Neuron Two:\n')
        f.write(np.array2string(initial_weights[1]))
        f.write('\n')
        f.write('Neuron Three:\n')
        f.write(np.array2string(initial_weights[2]))
        f.write('\n\n')
        f.write('Final weights: \n')
        f.write('Neuron One:\n')
        f.write(np.array2string(trained_weights[0]))
        f.write('\n')
        f.write('Neuron Two:\n')
        f.write(np.array2string(trained_weights[1]))
        f.write('\n')
        f.write('Neuron Three:\n')
        f.write(np.array2string(trained_weights[2]))
        f.write('\n\n')
        f.write('Training Targets:\n')
        f.write(target_train_string)
        f.write('\n')
        f.write('Predicted Training Targets:\n')
        f.write(predicted_train_target_string)
        f.write('\n\n')
        f.write('Testing Targets:\n')
        f.write(target_test_string)
        f.write('\n')
        f.write('Predicted Testing Targets:\n')
        f.write(predicted_test_target_string)
        f.write('\n\n')
        f.write('Number of Epochs to train the model:\n')
        f.write(str(num_epochs_elapsed))
        f.write('\n')
        f.write('Termination criteria:\n')
        f.write('Training model of 0.96 accuracy.\n\n')
        f.write('Confusion Matrix:\n')
        f.write(np.array2string(confusion_matrix_results))
        f.write('\n\n')
        f.write('Classification Report:\n')
        f.write(classification_report_results)
        f.write('\n')


    
    