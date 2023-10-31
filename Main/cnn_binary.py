import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

#Directories for training and testing data.
DATADIR = r'C:\Users\olath\programmer\IN5490\data_attack_elite_nonelite'
LABELS = ["elite", "nonelite"]
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

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X_TRAIN, Y_TRAIN, test_size=0.2, random_state=42)

X_TRAIN = np.array(X_TRAIN).reshape(-1, IMG_SIZE, IMG_SIZE,3)
Y_TRAIN = np.array(Y_TRAIN)
X_TEST = np.array(X_TEST).reshape(-1, IMG_SIZE, IMG_SIZE,3)
Y_TEST = np.array(Y_TEST)

X_TRAIN = X_TRAIN/255

#Building the model
model = Sequential()
model.add(Conv2D(32, (5,5), input_shape = X_TRAIN.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(32, (5,5)))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation("sigmoid"))

optimizer = Adam()
model.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_TRAIN,Y_TRAIN, epochs=11, validation_split=0.2, batch_size = 32, shuffle = True)

predictions = (model.predict(X_TEST)).round(decimals=0)
#print(Y_TEST)
#print(predictions)
print("Test accuracy: ",accuracy_score(Y_TEST, predictions))
cm = confusion_matrix(Y_TEST, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

