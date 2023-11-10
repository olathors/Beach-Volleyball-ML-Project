import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.image
import cv2 as cv
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score, precision_score, f1_score
import scipy.misc

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

plt.rcParams.update(tex_fonts)

#Directories for training and testing data.
DATADIR = r'C:\Users\olath\programmer\IN5490\dataset_eq_attack_4sec_overlap'
TESTDIR = r'C:\Users\olath\programmer\IN5490\dataset_eq_attack_4sec_overlap_test'
#RANDOMDIR = r'C:\Users\path' #Only for random testing
LABELS = ["nonelite", "elite"]
X_TRAIN = []
Y_TRAIN = []
X_TEST = []
Y_TEST = []
#X_RANDOM = []
#Y_RANDOM = []

IMG_SIZE = 128

#Creating the training set
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

#Creating the test set
for category in LABELS:
    path = os.path.join(TESTDIR, category)
    class_num = LABELS.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv.imread(os.path.join(path, img))
            new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
            X_TEST.append(new_array)
            Y_TEST.append(class_num)
        except Exception as e:
            pass

#Creating random test set
"""
for category in LABELS:
    path = os.path.join(RANDOMDIR, category)
    class_num = LABELS.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv.imread(os.path.join(path, img))
            new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
            X_RANDOM.append(new_array)
            Y_RANDOM.append(class_num)
        except Exception as e:
            pass
            """

X_TRAIN = np.array(X_TRAIN).reshape(-1, IMG_SIZE, IMG_SIZE,3)
Y_TRAIN = np.array(Y_TRAIN)
X_TEST = np.array(X_TEST).reshape(-1, IMG_SIZE, IMG_SIZE,3)
Y_TEST = np.array(Y_TEST)
#X_RANDOM = np.array(X_RANDOM).reshape(-1, IMG_SIZE, IMG_SIZE,3)
#Y_RANDOM = np.array(Y_RANDOM)



X_TRAIN = X_TRAIN/255


#Building the model
model = Sequential()
model.add(Conv2D(32, (5,5), input_shape = X_TRAIN.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activity_regularizer=tf.keras.regularizers.L1L2()))
model.add(Activation("sigmoid"))

optimizer = Adam(learning_rate=0.0002)
model.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics=tf.keras.metrics.Precision())
history = model.fit(X_TRAIN,Y_TRAIN, epochs=20, validation_split=0.1, batch_size = 32, shuffle = True)

predictions = (model.predict(X_TEST)).round(decimals=0)

print("Test accuracy: ",accuracy_score(Y_TEST, predictions))
print("Test recall: " ,recall_score(Y_TEST, predictions))
print("Test precision: ",precision_score(Y_TEST, predictions))
print("Test F1: " ,f1_score(Y_TEST, predictions))
      
cm = confusion_matrix(Y_TEST, predictions)
"""
predictions2 = (model.predict(X_RANDOM)).round(decimals=0)
#print(Y_TEST)
#print(predictions)
print("Test random accuracy: ",accuracy_score(Y_RANDOM, predictions2))
print("Test random recall: " ,recall_score(Y_RANDOM, predictions2))
print("Test random precision: ",precision_score(Y_RANDOM, predictions2))
print("Test random F1: " ,f1_score(Y_RANDOM, predictions2))
      
cm2 = confusion_matrix(Y_RANDOM, predictions2)
"""

fig, ax = plt.subplots(figsize=[3.4869240348692405, 3.4869240348692405])
ConfusionMatrixDisplay(confusion_matrix = cm).plot(ax = ax)
plt.show()
"""
fig, ax = plt.subplots(figsize=[3.4869240348692405, 3.4869240348692405])
ConfusionMatrixDisplay(confusion_matrix = cm2).plot(ax = ax)
plt.show()
"""

plt.figure(figsize=[3.4869240348692405, 3.4869240348692405], constrained_layout=True)
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('Model precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.figure(figsize=[3.4869240348692405, 3.4869240348692405], constrained_layout=True)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

