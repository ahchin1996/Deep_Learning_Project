import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as mat
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from keras import layers, optimizers, models
from sklearn.preprocessing import LabelEncoder

#讀取資料並做資料前處理將特殊字元處理掉
train_data =pd.read_csv('D:/Python/ML_report1/adult.data',sep=" ", header=None)
train_data = train_data.replace({'\$': '', ',': '','\.':'','<=50K':'1','>50K':'0'}, regex=True) #砍掉換行&逗號
train_data.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','label']

test_data =pd.read_csv('D:/Python/ML_report1/adult.test',sep=" ", header=None)
test_data = test_data.replace({'\$': '', ',': '','<=50K.':'1','>50K.':'0'}, regex=True) #砍掉換行&逗號
test_data.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','label']

#合併train data和test data
data_1 = train_data
data_1=data_1.append(test_data)
#data_1

#將缺值欄位補上平均值
data_1.replace('?', np.nan, inplace=True)
data_1=data_1.fillna(data_1.mean())
data_1 = data_1.apply(lambda x:x.fillna(x.value_counts().index[0]))
#data_1

#將資料轉成int
data_1[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','label'] ]=data_1[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','label']].astype(str).astype(int)
#將文字資料(要做one hot encoding和label切出)
data_cat = data_1[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']]
data_hours = data_1[['hours-per-week']]

#one hot encoding
data_cat = pd.get_dummies(data_cat)
#data_cat

#newdata = data_1.drop(['hours-per-week','workclass','education','marital-status','occupation','relationship','race','sex','native-country'],axis=1).join(data_cat,how='left')
#newdata

#將原始資料合併one hot encoding後的資料
newdata1 = data_1.drop(['workclass','education','marital-status','occupation','relationship','race','sex','native-country'],axis=1)
newdata_merge = pd.concat([newdata1,data_cat],axis=1).reindex(data_1.index)

#newdata_merge

#將資料正規化
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(newdata_merge)
data_norm= pd.DataFrame(np_scaled, columns = newdata_merge.columns)
data_norm.head()

#test_data=newdata_merge.iloc[32561:]
#test_data

#train_data=newdata_merge.iloc[:32561]
#train_data

test_data1=data_norm.iloc[32561:] #分解train data和test data
test_data1

train_data1=data_norm.iloc[:32561] #分解train data和test data
train_data1

train_true=data_hours.iloc[:32561] #擷取traindata label
#train_true

test_true=data_hours.iloc[32561:] #擷取testdata label
#test_true

from keras import layers, optimizers, models
from sklearn.preprocessing import LabelEncoder

X_train1 = train_data1.drop('hours-per-week', axis=1)
y_train1 = train_true

X_test1 = test_data1.drop('hours-per-week', axis=1)
y_test1 = test_true

from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense , Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping

#dense指最普通的全連接層型態
#dropout減少節點 通常設0.2~0.5 最高不會超過0.5
#2層(最後一層ouput不算)


early_stopping = EarlyStopping(monitor='val_loss', patience=200,restore_best_weights=True)
model = models.Sequential()
model.add(layers.Dense(16, input_shape=(X_train1.shape[1],), activation="relu"))
model.add(layers.Dropout(0.30)) #d砍上一層的節點
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train1, y_train1, validation_split=0.2, epochs=1000000, batch_size=128,callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test1, y_test1)
print("Test Acc : " + str(accuracy))
print("Test Loss : " + str(loss))

y_pred1 = model.predict(X_test1)
print('MSE為：',mean_squared_error(y_test1,y_pred1))
#print('MSE為(直接计算)：',np.mean((y_test-y_pred)**2))
print('RMSE為：',np.sqrt(mean_squared_error(y_test1,y_pred1)))

# from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers.core import Dense , Dropout
# from keras import regularizers
# from keras.callbacks import EarlyStopping
#
# #dense指最普通的全連接層型態
# #dropout減少節點 通常設0.2~0.5 最高不會超過0.5
# #2層(最後一層ouput不算)
#
#
# early_stopping = EarlyStopping(monitor='val_loss', patience=200,restore_best_weights=True)
# model = models.Sequential()
# model.add(layers.Dense(64, input_shape=(X_train1.shape[1],), activation="relu"))
# model.add(layers.Dropout(0.30)) #d砍上一層的節點
# model.add(layers.Dense(32, activation="relu"))
# model.add(layers.Dropout(0.30))
# model.add(layers.Dense(16, activation="relu"))
# model.add(layers.Dropout(0.30))
# model.add(layers.Dense(8, activation="relu"))
# model.add(layers.Dropout(0.30))
# model.add(layers.Dense(1))
#
# model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
#
# history = model.fit(X_train1, y_train1, validation_split=0.2, epochs=1000000, batch_size=128,callbacks=[early_stopping])
#
# loss, accuracy = model.evaluate(X_test1, y_test1)
# print("Test Acc : " + str(accuracy))
# print("Test Loss : " + str(loss))
# y_pred = model.predict(X_test1)
# print('MSE為：',mean_squared_error(y_test1,y_pred))
# #print('MSE為(直接计算)：',np.mean((y_test-y_pred)**2))
# print('RMSE為：',np.sqrt(mean_squared_error(y_test1,y_pred)))