import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

#Directories for training and testing data.
DATADIR = r'C:\Users\olath\programmer\IN5490\training_data_10sec'
TESTDIR = r'C:\Users\olath\programmer\IN5490\testing_data_10sec'
LABELS = ["spike", "nonspike"]
X_TRAIN = []
Y_TRAIN = []

IMG_SIZE = 465

for category in LABELS:
    path = os.path.join(DATADIR, category)
    class_num = LABELS.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv.imread(os.path.join(path, img))
            new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
            X_TRAIN.append(new_array)
            Y_TRAIN.append(class_num)
        except Exception as e:
            pass

print(len(X_TRAIN))

X_TRAIN = np.array(X_TRAIN).reshape(-1, IMG_SIZE, IMG_SIZE,3)
Y_TRAIN = np.array(Y_TRAIN)

X_TRAIN = X_TRAIN/255

#Building the model
model = Sequential()
model.add(Conv2D(32, (5,5), input_shape = X_TRAIN.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (5,5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (5,5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(optimizer='SGD', loss = 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_TRAIN,Y_TRAIN, epochs=30, validation_split=0.2)

for img in os.listdir(TESTDIR):
    try:
        img_array = cv.imread(os.path.join(TESTDIR, img))
        new_img = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_shape = new_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        predictions = model.predict(new_shape)
        plt.imshow(new_img)
        print("Predicted label: ", LABELS[round(predictions[0][0])], ' for image ', os.path.basename(img))
    except Exception as e:
        pass
