import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras import layers
from keras import models
import os
import tensorflow as tf
# 控制顯卡內核
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
sess0 = tf.compat.v1.InteractiveSession(config=config)

train_data =pd.read_csv('D:/Python/ML_report1/adult.data',sep=" ", header=None)
train_data = train_data.replace({'\$': '', ',': '','\.':'','<=50K':'1','>50K':'0'}, regex=True) #砍掉換行&逗號
train_data.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','label']

test_data =pd.read_csv('D:/Python/ML_report1/adult.test',sep=" ", header=None)
test_data = test_data.replace({'\$': '', ',': '','<=50K.':'1','>50K.':'0'}, regex=True) #砍掉換行&逗號
test_data.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','label']

data_1 = train_data
data_1=data_1.append(test_data)

data_1.replace('?', np.nan, inplace=True)
data_1=data_1.fillna(data_1.mean())
data_1 = data_1.apply(lambda x:x.fillna(x.value_counts().index[0]))

data_1[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','label'] ]=data_1[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','label']].astype(str).astype(int)
data_cat = data_1[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']]

data_cat = pd.get_dummies(data_cat)

newdata1 = data_1.drop(['workclass','education','marital-status','occupation','relationship','race','sex','native-country'],axis=1)
newdata_merge = pd.concat([newdata1,data_cat],axis=1).reindex(data_1.index)

newdata_merge

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(newdata_merge[:len(newdata_merge)-1])
data_norm= pd.DataFrame(np_scaled, columns = newdata_merge.columns)
data_norm.head()

test_data=data_norm.iloc[32561:]
test_data

train_data=data_norm.iloc[:32561]
train_data


X_train = train_data.drop('label', axis=1)
y_train = train_data['label']

X_test = test_data.drop('label', axis=1)
y_test = test_data['label']


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=200,restore_best_weights=True)
model = models.Sequential()
model.add(layers.Dense(64, input_shape=(X_train.shape[1],), activation="sigmoid"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(32, activation="sigmoid"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(16, activation="sigmoid"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(8, activation="sigmoid"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

model.compile(loss='binary_crossentropy', optimizer='rmsprop' ,metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=128,callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Acc : " + str(accuracy))
print("Test Loss : " + str(loss))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
#
# test_data1=newdata_merge.iloc[32561:]
# test_data1
#
# train_data1=newdata_merge.iloc[:32561]
# train_data1
#
# from keras import layers, optimizers, models
# from sklearn.preprocessing import LabelEncoder
#
# X_train1 = train_data1.drop('label', axis=1)
# y_train1 = train_data1['label']
#
# X_test1 = test_data1.drop('label', axis=1)
# y_test1 = test_data1['label']
#
# from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=200,restore_best_weights=True)
# model = models.Sequential()
# model.add(layers.Dense(64, input_shape=(X_train.shape[1],), activation="sigmoid"))
# model.add(layers.Dropout(0.30))
# model.add(layers.Dense(32, activation="sigmoid"))
# model.add(layers.Dropout(0.30))
# model.add(layers.Dense(16, activation="sigmoid"))
# model.add(layers.Dropout(0.30))
# model.add(layers.Dense(8, activation="sigmoid"))
# model.add(layers.Dropout(0.30))
# model.add(layers.Dense(1, activation="sigmoid"))
#
# #opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# model.compile(loss='binary_crossentropy', optimizer='adam' ,metrics=['accuracy'])
#
# history = model.fit(X_train, y_train, validation_split=0.2, epochs=1000000, batch_size=128,callbacks=[early_stopping])
#
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Test Acc : " + str(accuracy))
# print("Test Loss : " + str(loss))
#
# import matplotlib.pyplot as plt
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()

