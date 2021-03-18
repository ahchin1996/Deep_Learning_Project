# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:43:52 2020

@author: cooke
"""

# from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os 
import random
import cv2 
from imutils import paths
import pandas as pd
from keras.preprocessing.image import img_to_array

train_data = []
train_labels = []
tr_labels = pd.read_csv('D:/Python/ML_report4/C1-P1_Train Dev_fixed/train.csv',index_col = False)
train_labels = tr_labels['label'] 


train_dir = 'D:/Python/ML_report4/C1-P1_Train Dev_fixed/C1-P1_Train'
train_imagePaths = list(paths.list_images(train_dir))
random.seed(42)
random.shuffle(train_imagePaths)
random.seed(42)
random.shuffle(train_labels)


for imagePath in train_imagePaths:
    # 讀取照片數據，轉成一維
    image = cv2.imread(imagePath)
    img = cv2.resize(image,(224,224),interpolation = cv2.INTER_LINEAR)
    img = img_to_array(img)
    train_data.append(img)


test_data = []
test_labels = []

te_labels = pd.read_csv('D:/Python/ML_report4/C1-P1_Train Dev_fixed/dev.csv',index_col = False)
test_labels = te_labels['label'] 


test_dir = 'D:/Python/ML_report4/C1-P1_Train Dev_fixed/C1-P1_Dev'
test_imagePaths = list(paths.list_images(test_dir))
random.seed(42)
random.shuffle(test_imagePaths)
random.seed(42)
random.shuffle(test_labels)

#讀取test資料
for imagePath in test_imagePaths:
    # 讀取照片數據，轉成一維
    image = cv2.imread(imagePath)
    img = cv2.resize(image,(224,224),interpolation = cv2.INTER_LINEAR)
    img = img_to_array(img)
    test_data.append(img)


n_test_data=[]
new_test_dir = 'D:/Python/ML_report4/C1-P1_Test'
new_test_imagePaths = list(paths.list_images(new_test_dir))

for imagePath in new_test_imagePaths:
    # 讀取照片數據，轉成一維
    image = cv2.imread(imagePath)
    img = cv2.resize(image,(224,224),interpolation = cv2.INTER_LINEAR)
    img = img_to_array(img)
    n_test_data.append(img)


train_data = np.array(train_data, dtype="float32") / 255.0
train_labels = np.array(train_labels)
test_data = np.array(test_data, dtype="float32") / 255.0
test_labels = np.array(test_labels)

n_test_data = np.array(n_test_data, dtype="float32") / 255.0

for i in range(len(train_labels)):
    train_labels[i] = train_labels[i].replace("A","0")
    train_labels[i] = train_labels[i].replace("B","1")
    train_labels[i] = train_labels[i].replace("C","2")

for i in range(len(test_labels)):
    test_labels[i] = test_labels[i].replace("A","0")
    test_labels[i] = test_labels[i].replace("B","1")
    test_labels[i] = test_labels[i].replace("C","2")

# lb = LabelBinarizer()
# train_labels = lb.fit_transform(train_labels)
# test_labels = lb.transform(test_labels)

# 控制顯卡內核
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True

sess0 = tf.InteractiveSession(config=config)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam

# 建立網路模型結構
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape=(224,224,3),activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1,activation='softmax'))

# model = Sequential([
#     Conv2D(64, (3, 3), input_shape=(224, 224, 3), padding='same', activation='relu'),
#     Conv2D(64, (3, 3), activation='relu', padding='same'),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Dropout(0.25),

#     Conv2D(128, (3, 3), activation='relu', padding='same'),
#     Conv2D(128, (3, 3), activation='relu', padding='same', ),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Dropout(0.25),

#     Conv2D(256, (3, 3), activation='relu', padding='same', ),
#     Conv2D(256, (3, 3), activation='relu', padding='same', ),
#     Conv2D(256, (3, 3), activation='relu', padding='same', ),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Dropout(0.25),

#     Conv2D(512, (3, 3), activation='relu', padding='same', ),
#     Conv2D(512, (3, 3), activation='relu', padding='same', ),
#     Conv2D(512, (3, 3), activation='relu', padding='same', ),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Dropout(0.25),

#     Conv2D(512, (3, 3), activation='relu', padding='same', ),
#     Conv2D(512, (3, 3), activation='relu', padding='same', ),
#     Conv2D(512, (3, 3), activation='relu', padding='same', ),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Dropout(0.25),

#     Flatten(),
#     Dense(4096, activation='relu'),
#     Dense(4096, activation='relu'),
#     Dense(3, activation='softmax')
# ])

model.summary()

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1.0e-4), metrics=['accuracy'])

print(train_data.shape,test_data.shape,train_labels.shape,test_labels.shape)

model.fit(train_data,train_labels)

predictions = model.predict(n_test_data)


# history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=50, batch_size=16,verbose=2)

# loss, accuracy = model.evaluate(test_data, test_labels)

# print("Test Acc : " + str(accuracy))
# print("Test Loss : " + str(loss))

# import matplotlib.pyplot as plt
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()

# def show_train_history(history):
#     fig=plt.gcf()
#     fig.set_size_inches(16, 6)
#     plt.subplot(121)
#     plt.plot(history.history["acc"])
#     plt.plot(history.history["val_acc"])
#     plt.title("Train History")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.legend(["train", "validation"], loc="upper left")
#     plt.subplot(122)
#     plt.plot(history.history["loss"])
#     plt.plot(history.history["val_loss"])
#     plt.title("Train History")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend(["train", "validation"], loc="upper left")
#     plt.show()

# import matplotlib.pyplot as plt
# show_train_history(history)
