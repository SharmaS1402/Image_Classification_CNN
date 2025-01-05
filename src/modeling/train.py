import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Loading Datasets
X_train = np.loadtxt(
    "../../data/external/Image Classification CNN Keras Dataset/input.csv",
    delimiter=",",
)
Y_train = np.loadtxt(
    "../../data/external/Image Classification CNN Keras Dataset/labels.csv",
    delimiter=",",
)
X_test = np.loadtxt(
    "../../data/external/Image Classification CNN Keras Dataset/input_test.csv",
    delimiter=",",
)
Y_test = np.loadtxt(
    "../../data/external/Image Classification CNN Keras Dataset/labels_test.csv",
    delimiter=",",
)

# Reshaping datasets
X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)

X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)

X_train = X_train / 255.0
X_test = X_test / 255.0

print("Shape of X_train ", X_train.shape)
print("Shape of Y_train ", Y_train.shape)
print("Shape of X_test ", X_test.shape)
print("Shape of Y_test ", Y_test.shape)

# Any random image
idx = random.randint(0, len(X_train))
plt.imshow(X_train[idx, :])
plt.show()

# Model Architecture
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, Y_train, epochs=5, batch_size=64)

model.evaluate(X_test, Y_test)

# Making Predictions
idx2 = random.randint(0, len(Y_test))
plt.imshow(X_test[idx2, :])
plt.show()

y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))
y_pred = y_pred > 0.5

if y_pred == 0:
    pred = "dog"
else:
    pred = "cat"

print("Our model says : ", pred)
