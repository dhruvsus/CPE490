import numpy as np
import csv
import sys
from keras import models, layers
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)


class PeopleSet:
    def __init__(self):
        self.ids = None
        self.labels = None
        self.input_properties = None

    def __eq__(self, other):
        return ((type(other) == PeopleSet) and self.ids == other.ids
                and self.labels == other.labels
                and self.input_properties == other.input_properties)

    def __repr__(self):
        return ("PeopleSet({!r}, {!r}, {!r})".format(self.ids, self.labels,
                                                     self.input_properties))

    def fixInput(self, input_properties):
        sex = {'female': '0', 'male': '1'}
        embark = {'C': '0', 'S': '1', 'Q': '2', '': '3'}
        input_properties[:, 1] = np.vectorize(sex.__getitem__)(
            input_properties[:, 1])
        input_properties[:, 6] = np.vectorize(embark.__getitem__)(
            input_properties[:, 6])
        np.putmask(input_properties, input_properties == '', 0)
        return input_properties.astype(float)

    @staticmethod
    def Normalize_samples(toNormalize):
        return (toNormalize - np.mean(toNormalize, axis=0)) / np.std(
            toNormalize, axis=0)

    def populate(self, iterCSV):
        inter_csv_list = np.asarray(list(iterCSV))
        self.ids = inter_csv_list[
            1:, np.where(inter_csv_list == 'PassengerId')[1]].astype(int)
        self.labels = inter_csv_list[
            1:, np.where(inter_csv_list == 'Survived')[1]].astype(int)
        pClassIndex = np.where(inter_csv_list == 'Pclass')[1]
        inputProperties = [
            pClassIndex, pClassIndex + 2, pClassIndex + 3, pClassIndex + 4,
            pClassIndex + 5, pClassIndex + 7, pClassIndex + 9
        ]
        self.input_properties = np.hstack(
            inter_csv_list[1:, inputProperties]).T
        self.input_properties = self.fixInput(self.input_properties)
        self.input_properties = PeopleSet.Normalize_samples(
            self.input_properties)


with open(sys.argv[1]) as csvfile:
    iterCSV = csv.reader(csvfile)
    train = PeopleSet()
    train.populate(iterCSV)

iterCSV = csv.reader(sys.stdin.readlines())
test = PeopleSet()
test.populate(iterCSV)
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(7, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer='Adam',
    metrics=['mean_squared_error'])
"""def fold_test(train_data, train_targets, num_folds, num_epochs):
    num_samples = train_data.shape[0]
    val_size = num_samples // num_folds
    batch_size = (num_samples - val_size) // 20  # Up to 20 batches, ....
    batch_size = max(64, batch_size)  # but at least 64 samples, ...
    batch_size = min(num_samples, batch_size)  # and no more than num_samples
    mae_per_fold = []

    # Repeat test for each fold
    for fold in range(num_folds):
        val_data = train_data[fold * val_size:(fold + 1) * val_size]
        val_targets = train_targets[fold * val_size:(fold + 1) * val_size]

        partial_train_data = np.concatenate(
            [train_data[:fold * val_size], train_data[(fold + 1) * val_size:]],
            axis=0  # Concatenate along sample axis
        )

        partial_train_targets = np.concatenate(
            [
                train_targets[:fold * val_size],
                train_targets[(fold + 1) * val_size:]
            ],
            axis=0  # Concatenate along sample axis
        )

        model, history = build_and_train_model(
            partial_train_data, partial_train_targets, val_data, val_targets,
            num_epochs, batch_size)

        model.evaluate(val_data, val_targets, verbose=0)
        mae_per_fold.append(history)
    return mae_per_fold



mean = train_data.mean(
    axis=0)  # Average corresponding values across sample axis
train_data -= mean  # Shift so average is zero

std = train_data.std(axis=0)
train_data /= std  # Scale so std = 1

test_data = (test_data - mean) / std  # Test data uses train data parameters

mse_per_fold = np.asarray(fold_test(train_data, train_targets, 4, 10))
for history in mse_per_fold:
    plt.plot(range(1, len(history) + 1), history)
plt.xlabel('No of epochs')
plt.ylabel('Validation MSE')
plt.show()
np.savetxt(sys.stdout, mse_per_fold)
averageMSE = np.sum(mse_per_fold[:, 9]) / 4
print(averageMSE)
"""

train_data, train_labels = train.input_properties, train.labels
test_data = test.input_properties
model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=32,
    shuffle=True,
    verbose=0)
test.labels = model.predict(test_data)
test.labels[test.labels < 0.5] = 0
test.labels[test.labels >= 0.5] = 1
np.savetxt(
    sys.stdout,
    np.hstack((test.ids.astype(int), test.labels.astype(int))),
    fmt="%d,%d")
