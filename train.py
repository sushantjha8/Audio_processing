import librosa
import os
from sklearn.cross_validation import train_test_split
from keras.utils import to_categorical
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


DATA_PATH = "./audio/"


# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)



def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    #creating array for labels
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state)



def prepare_dataset(path=DATA_PATH):
    labels, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path  + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            # Downsampling
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data


def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]


# print(prepare_dataset(DATA_PATH))
X_train, X_test, y_train, y_test = get_train_test()

print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 20, 11, 1) #reshape for single length size
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
print(X_test.shape)
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
"""Model designing of speech to text """
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))#give dropout for each itration for over fitting
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))#give dropout for each itration for over fitting
model.add(Dense(7, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(X_train, y_train_hot, batch_size=100, epochs=200, verbose=1, validation_data=(X_test, y_test_hot))

""" This model provide 0.908 acc."""
"""Predicting Word"""
filefo=input("Enter wav file address       ")
sample = mfcc1(filefo)

# We need to reshape it remember?
sample_reshaped = sample.reshape(1, 20, 11, 1)

# Perform forward pass
print(get_labels()[0][
    np.argmax(model.predict(sample_reshaped))
])
"""Accuracy 0.95 or > give accurtae output"""
