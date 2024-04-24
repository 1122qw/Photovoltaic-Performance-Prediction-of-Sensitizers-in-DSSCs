from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# -----------------------data_reader---------------------------------------
fold = 10
test_size = 0.1
DF_data = pd.read_excel(r'E:\data\LHE.xlsx')
DF_data = np.array(DF_data)

# normalization
scaler = MinMaxScaler()
dt = scaler.fit_transform(DF_data[:, :44])

# PCA
x = dt
pca = PCA(n_components=16)  # dimensionality reduction
pca.fit(x)
# dimensionality reduction
x = pca.transform(x)

length_dt_x = dt.shape[0]
dt_x = x
dt_y = DF_data[:, -1]

dt_train_x, dt_test_x, dt_train_y, dt_test_y = train_test_split(dt_x, dt_y, random_state=24, test_size=test_size)

length_train_x = dt_train_x.shape[0]
data_train_length = length_train_x // fold

kfolder = KFold(n_splits=fold, shuffle=True, random_state=42)
kfold = kfolder.split(dt_train_x, dt_train_y)

j = 0
k_train_pearson = []
k_val_pearson = []
k_train_MSE = []
k_val_MSE = []
k_train_R2 = []
k_val_R2 = []

# ------------------------------------network--------------------------
for train_index, val_index in kfold:
    k_x_train = dt_train_x[train_index]  # training set
    k_y_train = dt_train_y[train_index]  # Tag of training set tag
    k_x_vali = dt_train_x[val_index]  # Testing set 
    k_y_vali = dt_train_y[val_index]  # Tag of testing set

    init = keras.initializers.he_normal(seed=42)
    model = Sequential()  # initialization
    model.add(Dense(256, activation='relu', input_shape=(16,), kernel_initializer=init))  # Get a fully connected layer
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', kernel_initializer=init))  # Get a fully connected layer
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_initializer=init))  # Get a fully connected layer
    model.add(Dense(128, activation='relu', kernel_initializer=init))  # Get a fully connected layer
    model.add(Dense(1, activation='linear'))  # Output layer

    filepath = r'E:\data\LHE.h5'
    # Once lifted, it covers once
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                                 mode='min', save_weights_only=False, period=1)
    callbacks_list = [checkpoint]
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=4e-7), loss='mse', metrics=['mae']) 
    model.fit(k_x_train.astype(np.float32), k_y_train.astype(np.float32), batch_size=32, epochs=500,
              validation_data=(k_x_vali.astype(np.float32), k_y_vali.astype(np.float32)), callbacks=callbacks_list)

    y_pred_train = model.predict(k_x_train.astype(np.float32))
    y_pred_valid = model.predict(k_x_vali.astype(np.float32))
    print(y_pred_train.shape, np.corrcoef(k_y_train.astype(np.float32).T, y_pred_train.T, rowvar=0).shape)
    model = model.save(r'E:\data\LHE.h5')
    print('No.' + str(j + 1) + ' cross validation')
    print('Pearson Coefficient of training set:', np.corrcoef(k_y_train.astype(np.float32), y_pred_train, rowvar=0)[0][1])  # r
    k_train_pearson.append(np.corrcoef(k_y_train.astype(np.float32), y_pred_train, rowvar=0)[0][1])
    print('Pearson Coefficient of testing set:', np.corrcoef(k_y_vali.astype(np.float32), y_pred_valid, rowvar=0)[0][1])  # r
    k_val_pearson.append(np.corrcoef(k_y_vali.astype(np.float32), y_pred_valid, rowvar=0)[0][1])
    print('MSE of training set:', mean_squared_error(k_y_train.astype(np.float32), y_pred_train))  # MSE
    k_train_MSE.append(mean_squared_error(k_y_train.astype(np.float32), y_pred_train))
    print('MSE of testing set:', mean_squared_error(k_y_vali.astype(np.float32), y_pred_valid))  # MSE
    k_val_MSE.append(mean_squared_error(k_y_vali.astype(np.float32), y_pred_valid))
    print('R2 of training set:', r2_score(k_y_train.astype(np.float32), y_pred_train))  # R2
    k_train_R2.append(r2_score(k_y_train.astype(np.float32), y_pred_train))
    print('R2 of testing set:', r2_score(k_y_vali.astype(np.float32), y_pred_valid))  # R2
    k_val_R2.append(r2_score(k_y_vali.astype(np.float32), y_pred_valid))

    j += 1

model = load_model(r'E:\data\LHE.h5')
y_pred_train = model.predict(dt_train_x.astype(np.float32))
y_pred_test = model.predict(dt_test_x.astype(np.float32))

train_score = []
test_score = []
print('Pearson Coefficient of training set:', np.corrcoef(dt_train_y.astype(np.float32), y_pred_train, rowvar=0)[0][1])
train_score.append(np.corrcoef(dt_train_y.astype(np.float32), y_pred_train, rowvar=0)[0][1])
print('Pearson Coefficient of testing set:', np.corrcoef(dt_test_y.astype(np.float32), y_pred_test, rowvar=0)[0][1])
test_score.append(np.corrcoef(dt_test_y.astype(np.float32), y_pred_test, rowvar=0)[0][1])
print('MSE of training set:', mean_squared_error(dt_train_y.astype(np.float32), y_pred_train))  # MSE
train_score.append(mean_squared_error(dt_train_y.astype(np.float32), y_pred_train))
print('MSE of testing set:', mean_squared_error(dt_test_y.astype(np.float32), y_pred_test))  # MSE
test_score.append(mean_squared_error(dt_test_y.astype(np.float32), y_pred_test))
print('R2 of training set:', r2_score(dt_train_y.astype(np.float32), y_pred_train))  # R2
train_score.append(r2_score(dt_train_y.astype(np.float32), y_pred_train))
print('R2 of testing set:', r2_score(dt_test_y.astype(np.float32), y_pred_test))  # R2
test_score.append(r2_score(dt_test_y.astype(np.float32), y_pred_test))

k_score = np.concatenate((
    np.array(k_train_pearson).reshape(1, fold),
    np.array(k_val_pearson).reshape(1, fold),
    np.array(k_train_MSE).reshape(1, fold),
    np.array(k_val_MSE).reshape(1, fold),
    np.array(k_train_R2).reshape(1, fold),
    np.array(k_val_R2).reshape(1, fold)), axis=0)

all_score = np.concatenate((
np.array(train_score).reshape(1, 3),
np.array(test_score).reshape(1, 3)), axis=0)
k_score = pd.DataFrame(k_score)
all_score = pd.DataFrame(all_score)
dt_train_y = pd.DataFrame(dt_train_y)
y_pred_train = pd.DataFrame(y_pred_train)
dt_test_y = pd.DataFrame(dt_test_y)
y_pred_test = pd.DataFrame(y_pred_test)
train = pd.concat([dt_train_y], axis=1)
test = pd.concat([dt_test_y], axis=1)

train.to_excel(r'E:\data\train.xlsx', header=False, index=False)
test.to_excel(r'E:\data\test.xlsx', header=False, index=False)
k_score.to_excel(r'E:\data\k_score.xlsx', header=False, index=False)
all_score.to_excel(r'E:\data\score.xlsx', header=False, index=False)
y_pred_train.to_excel(r'E:\data\y_pred_train.xlsx', header=False, index=False)
y_pred_test.to_excel(r'E:\data\y_pred_test.xlsx', header=False, index=False)