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
DF_data = pd.read_excel(r'E:\data\1-result_data.xlsx')
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
dt_y = DF_data[:,44:]

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
i = 0
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
    model.add(Dense(23, activation='linear'))   # Output layer

    filepath = r'E:\data\all-1.h5'
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
    model = model.save(r'E:\data\all-1.h5')
    print('No.' + str(j + 1) + ' cross validation')
    sum_train_pearson = 0
    sum_train_mse = 0
    sum_train_r2 = 0

    for i in range(y_pred_train.shape[0]):
        y1 = k_y_train.astype(np.float32)[i]
        y2 = y_pred_train[i]
        pearson_train1 = np.corrcoef(y1, y2, rowvar=0)[0][1]
        mse_train1 = mean_squared_error(y1, y2)
        r2_train1 = r2_score(y1, y2)

        sum_train_pearson += pearson_train1
        sum_train_mse += mse_train1
        sum_train_r2 += r2_train1

    sum_train_pearson = sum_train_pearson / y_pred_train.shape[0]
    sum_train_mse = sum_train_mse / y_pred_train.shape[0]
    sum_train_r2 = sum_train_r2 / y_pred_train.shape[0]

    print(sum_train_pearson, sum_train_mse, sum_train_r2)
    sum_val_pearson = 0
    sum_val_mse = 0
    sum_val_r2 = 0

    for i in range(y_pred_valid.shape[0]):
        y1 = k_y_vali.astype(np.float32)[i]
        y2 = y_pred_valid[i]
        pearson_val1 = np.corrcoef(y1, y2, rowvar=0)[0][1]
        mse_val1 = mean_squared_error(y1, y2)
        r2_val1 = r2_score(y1, y2)

        sum_val_pearson += pearson_val1
        sum_val_mse += mse_val1
        sum_val_r2 += r2_val1

    sum_val_pearson = sum_val_pearson / y_pred_valid.shape[0]
    sum_val_mse = sum_val_mse / y_pred_valid.shape[0]
    sum_val_r2 = sum_val_r2 / y_pred_valid.shape[0]
    print(sum_val_pearson, sum_val_mse, sum_val_r2)

    k_train_pearson.append(sum_train_pearson)
    k_val_pearson.append(sum_val_pearson)
    k_train_MSE.append(sum_train_mse)
    k_val_MSE.append(sum_val_mse)
    k_train_R2.append(sum_train_r2)
    k_val_R2.append(sum_val_r2)

    i += 1
    j += 1

model = load_model(r'E:\data\all-1.h5')
y_pred_train = model.predict(dt_train_x.astype(np.float32))
y_pred_test = model.predict(dt_test_x.astype(np.float32))

train_score = []
test_score = []

sum_test_pearson = 0
sum_test_mse = 0
sum_test_r2 = 0

for i in range(y_pred_test.shape[0]):
    y1 = dt_test_y.astype(np.float32)[i]
    y2 = y_pred_test[i]
    pearson_test1 = np.corrcoef(y1, y2, rowvar=0)[0][1]
    mse_test1 = mean_squared_error(y1, y2)
    r2_test1 = r2_score(y1, y2)

    sum_test_pearson += pearson_test1
    sum_test_mse += mse_test1
    sum_test_r2 += r2_test1

sum_test_pearson = sum_test_pearson / y_pred_test.shape[0]
sum_test_mse = sum_test_mse / y_pred_test.shape[0]
sum_test_r2 = sum_test_r2 / y_pred_test.shape[0]
print(sum_test_pearson, sum_test_mse, sum_test_r2)

k_score = np.concatenate((
    np.array(k_train_pearson).reshape(1, fold),
    np.array(k_val_pearson).reshape(1, fold),
    np.array(k_train_MSE).reshape(1, fold),
    np.array(k_val_MSE).reshape(1, fold),
    np.array(k_train_R2).reshape(1, fold),
    np.array(k_val_R2).reshape(1, fold)), axis=0)

all_score = np.concatenate((
np.array(sum_test_pearson).reshape(1, 1),
np.array(sum_test_mse).reshape(1, 1),
np.array(sum_test_r2).reshape(1, 1)), axis=1)

k_score = pd.DataFrame(k_score)
all_score = pd.DataFrame(all_score)
dt_train_y = pd.DataFrame(dt_train_y)
y_pred_train = pd.DataFrame(y_pred_train)
dt_test_y = pd.DataFrame(dt_test_y)
y_pred_test = pd.DataFrame(y_pred_test)
train = pd.concat([dt_train_y], axis=1)
test = pd.concat([dt_test_y], axis=1)

train.to_excel(r'E:\data\train.xlsx',  header=False, index=False)
test.to_excel(r'E:\data\test.xlsx',  header=False, index=False)
k_score.to_excel(r'E:\data\k_score.xlsx',  header=False, index=False)
all_score.to_excel(r'E:\data\score.xlsx',  header=False, index=False)
y_pred_train.to_excel(r'E:\data\y_pred_train.xlsx',  header=False, index=False)
y_pred_test.to_excel(r'E:\data\y_pred_test.xlsx', header=False, index=False)