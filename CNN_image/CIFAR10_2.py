from sklearn.preprocessing import LabelBinarizer
import numpy as np
import random
import cv2
import os
from imutils import paths

#讀取train資料
train_data = []
train_labels = []
train_dir = 'D:/Python/ML_report2/CIFAR10/train'
train_imagePaths = list(paths.list_images(train_dir))
random.seed(42)
random.shuffle(train_imagePaths)

for imagePath in train_imagePaths:
    # 讀取照片數據，轉成一維
    image = cv2.imread(imagePath)
    train_data.append(image)

    # 讀取標籤
    label = imagePath.split(os.path.sep)[-2]
    train_labels.append(label)

test_data = []
test_labels = []
test_dir = 'D:/Python/ML_report2/CIFAR10/test'
test_imagePaths = list(paths.list_images(test_dir))
random.seed(42)
random.shuffle(test_imagePaths)


#讀取test資料
for imagePath in test_imagePaths:
    # 讀取照片數據，轉成一維
    image = cv2.imread(imagePath)
    test_data.append(image)

    # 讀取標籤
    label = imagePath.split(os.path.sep)[-2]
    test_labels.append(label)

train_data = np.array(train_data, dtype="float32") / 255.0
train_labels = np.array(train_labels)
test_data = np.array(test_data, dtype="float32") / 255.0
test_labels = np.array(test_labels)

lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)

# 控制顯卡內核
import tensorflow as tf
# 控制顯卡內核
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True

sess0 = tf.compat.v1.InteractiveSession(config=config)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam
# 建立網路模型結構


model = Sequential([
    Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu', padding='same', ),
    Conv2D(256, (3, 3), activation='relu', padding='same', ),
    Conv2D(256, (3, 3), activation='relu', padding='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.25),

    Conv2D(512, (3, 3), activation='relu', padding='same', ),
    Conv2D(512, (3, 3), activation='relu', padding='same', ),
    Conv2D(512, (3, 3), activation='relu', padding='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.25),

    Conv2D(512, (3, 3), activation='relu', padding='same', ),
    Conv2D(512, (3, 3), activation='relu', padding='same', ),
    Conv2D(512, (3, 3), activation='relu', padding='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1.0e-4), metrics=['accuracy'])

print(train_data.shape,test_data.shape,train_labels.shape,test_labels.shape)

history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=50, batch_size=128,verbose=2)

loss, accuracy = model.evaluate(test_data, test_labels)

print("Test Acc : " + str(accuracy))
print("Test Loss : " + str(loss))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

def show_train_history(history):
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

import matplotlib.pyplot as plt
show_train_history(history)