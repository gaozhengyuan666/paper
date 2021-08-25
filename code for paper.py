import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dropout
from keras.models import Sequential
from tensorflow import optimizers
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

import warnings

warnings.filterwarnings('ignore')

'''
prepare for training
'''
## load data
sh = pd.read_csv('/content/gdrive/MyDrive/gzy paper/data/000001.csv', parse_dates=True, index_col=0)
sz = pd.read_csv('/content/gdrive/MyDrive/gzy paper/data/399001.csv', parse_dates=True, index_col=0)

## new features for sh
# MACD
ema_12 = sh['Close'].ewm(span=12, min_periods=12).mean()
ema_26 = sh['Close'].ewm(span=26, min_periods=26).mean()
dif = ema_12 - ema_26
dea = dif.ewm(span=9, min_periods=9).mean()
macd = (dif - dea) * 2
sh['MACD'] = macd
# stochatic k
sto_k = (sh['Close'] - sh['Low']) / (sh['High'] - sh['Low']) * 100
sh['% K'] = sto_k
sh.dropna(inplace=True)
sh = sh[['Close', 'Volume', '% K', 'MACD']]
sh.columns = ['sh', 'sh volume', 'sh %k', 'sh macd']
sh = sh[sh.index >= '2007-01-01']

## new features for sz
# MACD
ema_12 = sz['Close'].ewm(span=12, min_periods=12).mean()
ema_26 = sz['Close'].ewm(span=26, min_periods=26).mean()
dif = ema_12 - ema_26
dea = dif.ewm(span=9, min_periods=9).mean()
macd = (dif - dea) * 2
sz['MACD'] = macd
# stochatic k
sto_k = (sz['Close'] - sz['Low']) / (sz['High'] - sz['Low']) * 100
sz['% K'] = sto_k
sz.dropna(inplace=True)
sz = sz[['Close', 'Volume', '% K', 'MACD']]
sz.columns = ['sz', 'sz volume', 'sz %k', 'sz macd']
sz = sz[sz.index >= '2007-01-01']

## split train,val,test for single-output model
# split1
sh_train1 = sh[(sh.index >= '2007-01-01') * (sh.index < '2019-07-01')]
sh_val1 = sh[(sh.index >= '2019-07-01') * (sh.index < '2019-10-01')]
sh_test1 = sh[(sh.index >= '2019-10-01') * (sh.index < '2020-01-01')]

# split2
sh_train2 = sh[(sh.index >= '2007-07-01') * (sh.index < '2020-01-01')]
sh_val2 = sh[(sh.index >= '2020-01-01') * (sh.index < '2020-04-01')]
sh_test2 = sh[(sh.index >= '2020-04-01') * (sh.index < '2020-07-01')]

# split3
sh_train3 = sh[(sh.index >= '2008-01-01') * (sh.index < '2020-07-01')]
sh_val3 = sh[(sh.index >= '2020-07-01') * (sh.index < '2020-10-01')]
sh_test3 = sh[(sh.index >= '2020-10-01') * (sh.index < '2021-01-01')]

# split4
sh_train4 = sh[(sh.index >= '2008-07-01') * (sh.index < '2021-01-01')]
sh_val4 = sh[(sh.index >= '2021-01-01') * (sh.index < '2021-04-01')]
sh_test4 = sh[(sh.index >= '2021-04-01') * (sh.index < '2021-07-01')]

print('shape1:', sh_train1.shape[0], sh_val1.shape[0], sh_test1.shape[0])
print('shape2:', sh_train2.shape[0], sh_val2.shape[0], sh_test2.shape[0])
print('shape3:', sh_train3.shape[0], sh_val3.shape[0], sh_test3.shape[0])
print('shape4:', sh_train4.shape[0], sh_val4.shape[0], sh_test4.shape[0])

shsz = pd.concat([pd.DataFrame({'sh': sh['sh'], 'sz': sz['sz']}), sh.iloc[:, 1:], sz.iloc[:, 1:]], axis=1)
## split train,val,test for multi-output model
# split1
shsz_train1 = shsz[(shsz.index >= '2007-01-01') * (shsz.index < '2019-07-01')]
shsz_val1 = shsz[(shsz.index >= '2019-07-01') * (shsz.index < '2019-10-01')]
shsz_test1 = shsz[(shsz.index >= '2019-10-01') * (shsz.index < '2020-01-01')]

# split2
shsz_train2 = shsz[(shsz.index >= '2007-07-01') * (shsz.index < '2020-01-01')]
shsz_val2 = shsz[(shsz.index >= '2020-01-01') * (shsz.index < '2020-04-01')]
shsz_test2 = shsz[(shsz.index >= '2020-04-01') * (shsz.index < '2020-07-01')]

# split3
shsz_train3 = shsz[(shsz.index >= '2008-01-01') * (shsz.index < '2020-07-01')]
shsz_val3 = shsz[(shsz.index >= '2020-07-01') * (shsz.index < '2020-10-01')]
shsz_test3 = shsz[(shsz.index >= '2020-10-01') * (shsz.index < '2021-01-01')]

# split4
shsz_train4 = shsz[(shsz.index >= '2008-07-01') * (shsz.index < '2021-01-01')]
shsz_val4 = shsz[(shsz.index >= '2021-01-01') * (shsz.index < '2021-04-01')]
shsz_test4 = shsz[(shsz.index >= '2021-04-01') * (shsz.index < '2021-07-01')]

print('shape1:', shsz_train1.shape[0], shsz_val1.shape[0], shsz_test1.shape[0])
print('shape2:', shsz_train2.shape[0], shsz_val2.shape[0], shsz_test2.shape[0])
print('shape3:', shsz_train3.shape[0], shsz_val3.shape[0], shsz_test3.shape[0])
print('shape4:', shsz_train4.shape[0], shsz_val4.shape[0], shsz_test4.shape[0])

'''
define functions
'''


# scaled for split
def scaling(train, val, test):
    sc1 = StandardScaler()
    train_scaled1 = sc1.fit_transform(train[['sh']])
    val_scaled1 = sc1.transform(val[['sh']])
    test_scaled1 = sc1.transform(test[['sh']])

    sc2 = StandardScaler()
    train_scaled2 = sc2.fit_transform(train.iloc[:, 1:])
    val_scaled2 = sc2.transform(val.iloc[:, 1:])
    test_scaled2 = sc2.transform(test.iloc[:, 1:])

    train_scaled = np.concatenate([train_scaled1, train_scaled2], axis=1)
    val_scaled = np.concatenate([val_scaled1, val_scaled2], axis=1)
    test_scaled = np.concatenate([test_scaled1, test_scaled2], axis=1)
    return train_scaled, val_scaled, test_scaled, sc1, sc2


# Setting up an early stop
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80, verbose=1, mode='min')
callbacks_list = [earlystop]


# Build and train the model
def fit_model_rnn(train, val, lag, hl, lr, batch_size, num_epochs):
    # Loop for training data
    X_train = []
    Y_train = []
    for i in range(lag, train.shape[0]):
        X_train.append(train[i - lag:i])
        Y_train.append(train[i][0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)

    # Adding layers to the model
    model = Sequential()
    model.add(
        SimpleRNN(hl[0], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    for i in range(1, len(hl) - 1):
        model.add(SimpleRNN(hl[i], return_sequences=True, activation='relu'))
        model.add(Dropout(0.5))
    model.add(SimpleRNN(hl[-1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mean_squared_error')

    # Training the data
    if val.shape[0] == 0:
        history = model.fit(X_train, Y_train, batch_size, num_epochs, validation_split=0.2, shuffle=False,
                            callbacks=callbacks_list, verbose=0)
        X_val, Y_val = [], []
    else:
        X_val = []
        Y_val = []
        for i in range(lag, val.shape[0]):
            X_val.append(val[i - lag:i])
            Y_val.append(val[i][0])
        X_val, Y_val = np.array(X_val), np.array(Y_val)
        history = model.fit(X_train, Y_train, batch_size, num_epochs, validation_data=(X_val, Y_val), shuffle=False,
                            callbacks=callbacks_list, verbose=0)
    model.reset_states()
    return model, history.history['loss'], history.history['val_loss']


def fit_model_lstm(train, val, lag, hl, lr, batch_size, num_epochs):
    # Loop for training data
    X_train = []
    Y_train = []
    for i in range(lag, train.shape[0]):
        X_train.append(train[i - lag:i])
        Y_train.append(train[i][0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)

    # Adding layers to the model
    model = Sequential()
    model.add(LSTM(hl[0], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    for i in range(1, len(hl) - 1):
        model.add(LSTM(hl[i], return_sequences=True, activation='relu'))
        model.add(Dropout(0.5))
    model.add(LSTM(hl[-1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mean_squared_error')

    # Training the data
    if val.shape[0] == 0:
        history = model.fit(X_train, Y_train, batch_size, num_epochs, validation_split=0.2, shuffle=False,
                            callbacks=callbacks_list, verbose=0)
        X_val, Y_val = [], []
    else:
        X_val = []
        Y_val = []
        for i in range(lag, val.shape[0]):
            X_val.append(val[i - lag:i])
            Y_val.append(val[i][0])
        X_val, Y_val = np.array(X_val), np.array(Y_val)
        history = model.fit(X_train, Y_train, batch_size, num_epochs, validation_data=(X_val, Y_val), shuffle=False,
                            callbacks=callbacks_list, verbose=0)
    model.reset_states()
    return model, history.history['loss'], history.history['val_loss']


def fit_model_gru(train, val, lag, hl, lr, batch_size, num_epochs):
    # Loop for training data
    X_train = []
    Y_train = []
    for i in range(lag, train.shape[0]):
        X_train.append(train[i - lag:i])
        Y_train.append(train[i][0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)

    # Adding layers to the model
    model = Sequential()
    model.add(GRU(hl[0], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    for i in range(1, len(hl) - 1):
        model.add(GRU(hl[i], return_sequences=True, activation='relu'))
        model.add(Dropout(0.5))
    model.add(GRU(hl[-1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mean_squared_error')

    # Training the data
    if val.shape[0] == 0:
        history = model.fit(X_train, Y_train, batch_size, num_epochs, validation_split=0.2, shuffle=False,
                            callbacks=callbacks_list, verbose=0)
        X_val, Y_val = [], []
    else:
        X_val = []
        Y_val = []
        for i in range(lag, val.shape[0]):
            X_val.append(val[i - lag:i])
            Y_val.append(val[i][0])
        X_val, Y_val = np.array(X_val), np.array(Y_val)
        history = model.fit(X_train, Y_train, batch_size, num_epochs, validation_data=(X_val, Y_val), shuffle=False,
                            callbacks=callbacks_list, verbose=0)
    model.reset_states()
    return model, history.history['loss'], history.history['val_loss']


# Evaluating the model
def evaluate_model(model, test, lag):
    X_test = []
    Y_test = []
    for i in range(lag, test.shape[0]):
        X_test.append(test[i - lag:i])
        Y_test.append(test[i][0])
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    predicted = model.predict(X_test)
    mse = mean_squared_error(Y_test, predicted)
    rmse = sqrt(mse)
    r2 = r2_score(Y_test, predicted)
    return mse, rmse, r2, Y_test, predicted


# Plotting the trainng errors
def plot_error(train_loss, val_loss):
    plt.plot(train_loss, c='b')
    plt.plot(val_loss, c='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss and Validation Loss Curve')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


# Plotting the prediction
def plot_data(Y_test, Y_hat):
    plt.plot(Y_test, c='b')
    plt.plot(Y_hat, c='r')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction using Multivariate-RNN')
    plt.legend(['Actual', 'Predicted'], loc='lower right')
    plt.show()


# scaled for split
def scaling_shsz(train, val, test):
    sc1 = StandardScaler()
    train_scaled1 = sc1.fit_transform(train[['sh']])
    val_scaled1 = sc1.transform(val[['sh']])
    test_scaled1 = sc1.transform(test[['sh']])

    sc2 = StandardScaler()
    train_scaled2 = sc2.fit_transform(train[['sz']])
    val_scaled2 = sc2.transform(val[['sz']])
    test_scaled2 = sc2.transform(test[['sz']])

    sc3 = StandardScaler()
    train_scaled3 = sc3.fit_transform(train.iloc[:, 2:])
    val_scaled3 = sc3.transform(val.iloc[:, 2:])
    test_scaled3 = sc3.transform(test.iloc[:, 2:])

    train_scaled = np.concatenate([train_scaled1, train_scaled2, train_scaled3], axis=1)
    val_scaled = np.concatenate([val_scaled1, val_scaled2, val_scaled3], axis=1)
    test_scaled = np.concatenate([test_scaled1, test_scaled2, test_scaled3], axis=1)
    return train_scaled, val_scaled, test_scaled, sc1, sc2, sc3


# Setting up an early stop
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80, verbose=1, mode='min')
callbacks_list = [earlystop]


# Build and train the model
def fit_model_shsz_gru(train, val, lag, hl, lr, batch_size, num_epochs):
    # Loop for training data
    X_train = []
    Y_train = []
    for i in range(lag, train.shape[0]):
        X_train.append(train[i - lag:i])
        Y_train.append(train[i][0:2])
    X_train, Y_train = np.array(X_train), np.array(Y_train)

    # Adding layers to the model
    model = Sequential()
    model.add(GRU(hl[0], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    for i in range(1, len(hl) - 1):
        model.add(GRU(hl[i], return_sequences=True, activation='relu'))
        model.add(Dropout(0.5))
    model.add(GRU(hl[-1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mean_squared_error')

    # Training the data
    if val.shape[0] == 0:
        history = model.fit(X_train, Y_train, batch_size, num_epochs, validation_split=0.2, shuffle=False,
                            callbacks=callbacks_list, verbose=0)
        X_val, Y_val = [], []
    else:
        X_val = []
        Y_val = []
        for i in range(lag, val.shape[0]):
            X_val.append(val[i - lag:i])
            Y_val.append(val[i][0])
        X_val, Y_val = np.array(X_val), np.array(Y_val)
        history = model.fit(X_train, Y_train, batch_size, num_epochs, validation_data=(X_val, Y_val), shuffle=False,
                            callbacks=callbacks_list, verbose=0)
    model.reset_states()
    return model, history.history['loss'], history.history['val_loss']


# Evaluating the model
def evaluate_model_shsz(model, test, lag):
    X_test = []
    Y_test = []
    for i in range(lag, test.shape[0]):
        X_test.append(test[i - lag:i])
        Y_test.append(test[i][0:2])
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    predicted = model.predict(X_test)
    mse = mean_squared_error(Y_test[:, 0], predicted[:, 0])
    rmse = sqrt(mse)
    r2 = r2_score(Y_test[:, 0], predicted[:, 0])
    return mse, rmse, r2, Y_test, predicted[:, 0], predicted[:, 1]


'''
single-output model (take split 1 as sample)
'''
### rnn
train_scaled, val_scaled, test_scaled, sc1, sc2 = scaling(sh_train1, sh_val1, sh_test1)
rmse_rnn1 = []
r2_rnn1 = []
predict_rnn1 = pd.DataFrame()
for i in range(5):
    # Hyperparamters
    lag = 5
    hl = [32, 32]
    lr = 1e-4
    batch_size = 64
    num_epochs = 150
    model_rnn1, train_loss_rnn1, val_loss_rnn1 = fit_model_rnn(train_scaled, val_scaled, lag, hl, lr, batch_size,
                                                               num_epochs)
    mse_train_rnn1, rmse_train_rnn1, r2_train_rnn1, Y_train_rnn1, predicted_train_rnn1 = evaluate_model(model_rnn1,
                                                                                                        train_scaled,
                                                                                                        lag)
    mse_val_rnn1, rmse_val_rnn1, r2_val_rnn1, Y_val_rnn1, predicted_val_rnn1 = evaluate_model(model_rnn1, val_scaled,
                                                                                              lag)
    mse_test_rnn1, rmse_test_rnn1, r2_test_rnn1, Y_test_rnn1, predicted_test_rnn1 = evaluate_model(model_rnn1,
                                                                                                   test_scaled, lag)
    rmse_rnn1.append(rmse_test_rnn1)
    r2_rnn1.append(r2_test_rnn1)
    predict_rnn1[str(i + 1)] = predicted_test_rnn1.flatten()
for i in range(5):
    predict_rnn1[str(i + 1) + '_price'] = sc1.inverse_transform(predict_rnn1[str(i + 1)])
predict_rnn1['avg_price'] = predict_rnn1.iloc[:, -5:].mean(axis=1)
predict_rnn1.to_csv('predict_rnn1.csv')
print(rmse_rnn1)
print(r2_rnn1)

### lstm
train_scaled, val_scaled, test_scaled, sc1, sc2 = scaling(sh_train1, sh_val1, sh_test1)
rmse_lstm1 = []
r2_lstm1 = []
predict_lstm1 = pd.DataFrame()
for i in range(5):
    # Hyperparamters
    lag = 5
    hl = [32, 32]
    lr = 1e-4
    batch_size = 64
    num_epochs = 150
    model_lstm1, train_loss_lstm1, val_loss_lstm1 = fit_model_lstm(train_scaled, val_scaled, lag, hl, lr, batch_size,
                                                                   num_epochs)
    mse_test_lstm1, rmse_test_lstm1, r2_test_lstm1, Y_test_lstm1, predicted_test_lstm1 = evaluate_model(model_lstm1,
                                                                                                        test_scaled,
                                                                                                        lag)
    rmse_lstm1.append(rmse_test_lstm1)
    r2_lstm1.append(r2_test_lstm1)
    predict_lstm1[str(i + 1)] = predicted_test_lstm1.flatten()
for i in range(5):
    predict_lstm1[str(i + 1) + '_price'] = sc1.inverse_transform(predict_lstm1[str(i + 1)])
predict_lstm1['avg_price'] = predict_lstm1.iloc[:, -5:].mean(axis=1)
predict_lstm1.to_csv('predict_lstm1.csv')
print(rmse_lstm1)
print(r2_lstm1)

### gru
train_scaled, val_scaled, test_scaled, sc1, sc2 = scaling(sh_train1, sh_val1, sh_test1)
rmse_gru1 = []
r2_gru1 = []
predict_gru1 = pd.DataFrame()
for i in range(5):
    # Hyperparamters
    lag = 5
    hl = [32, 32]
    lr = 1e-4
    batch_size = 64
    num_epochs = 150
    model_gru1, train_loss_gru1, val_loss_gru1 = fit_model_gru(train_scaled, val_scaled, lag, hl, lr, batch_size,
                                                               num_epochs)
    mse_test_gru1, rmse_test_gru1, r2_test_gru1, Y_test_gru1, predicted_test_gru1 = evaluate_model(model_gru1,
                                                                                                   test_scaled, lag)
    rmse_gru1.append(rmse_test_gru1)
    r2_gru1.append(r2_test_gru1)
    predict_gru1[str(i + 1)] = predicted_test_gru1.flatten()
for i in range(5):
    predict_gru1[str(i + 1) + '_price'] = sc1.inverse_transform(predict_gru1[str(i + 1)])
predict_gru1['avg_price'] = predict_gru1.iloc[:, -5:].mean(axis=1)
predict_gru1.to_csv('predict_gru1.csv')
print(rmse_gru1)
print(r2_gru1)

'''
multi-output model (take split 1 as sample)
'''
train_scaled, val_scaled, test_scaled, sc1, sc2, sc3 = scaling_shsz(shsz_train1, shsz_val1, shsz_test1)
rmse_multi_gru1 = []
r2_multi_gru1 = []
sh_predict_multi_gru1 = pd.DataFrame()
sz_predict_multi_gru1 = pd.DataFrame()
for i in range(5):
    # Hyperparamters
    lag = 5
    hl = [32, 32]
    lr = 1e-4
    batch_size = 64
    num_epochs = 150
    model_multi_gru1, train_loss_multi_gru1, val_loss_multi_gru1 = fit_model_shsz_gru(train_scaled, val_scaled, lag, hl,
                                                                                      lr, batch_size, num_epochs)
    mse_test_multi_gru1, rmse_test_multi_gru1, r2_test_multi_gru1, Y_test_multi_gru1, sh_predicted_test_multi_gru1, sz_predicted_test_multi_gru1 = evaluate_model_shsz(
        model_multi_gru1, test_scaled, lag)
    rmse_multi_gru1.append(rmse_test_multi_gru1)
    r2_multi_gru1.append(r2_test_multi_gru1)
    sh_predict_multi_gru1[str(i + 1)] = sh_predicted_test_multi_gru1.flatten()
    sz_predict_multi_gru1[str(i + 1)] = sz_predicted_test_multi_gru1.flatten()
for i in range(5):
    sh_predict_multi_gru1[str(i + 1) + '_price'] = sc1.inverse_transform(sh_predict_multi_gru1[str(i + 1)])
    sz_predict_multi_gru1[str(i + 1) + '_price'] = sc2.inverse_transform(sz_predict_multi_gru1[str(i + 1)])
sh_predict_multi_gru1['avg_price'] = sh_predict_multi_gru1.iloc[:, -5:].mean(axis=1)
sz_predict_multi_gru1['avg_price'] = sz_predict_multi_gru1.iloc[:, -5:].mean(axis=1)
sh_predict_multi_gru1.to_csv('sh_predict_multi_gru1.csv')
sz_predict_multi_gru1.to_csv('sz_predict_multi_gru1.csv')
print(rmse_multi_gru1)
print(r2_multi_gru1)

'''
plot (take split 1 as sample)
'''
rnn1 = pd.read_csv('predict_rnn1.csv')['avg_price']
lstm1 = pd.read_csv('predict_lstm1.csv')['avg_price']
gru1 = pd.read_csv('predict_gru1.csv')['avg_price']
actual1 = sh_test1['sh'][5:]
rnn1.index = actual1.index
lstm1.index = actual1.index
gru1.index = actual1.index

plt.figure(figsize=(20, 20), dpi=200)
plt.plot(rnn1, color='red', label='RNN', linestyle='--')
plt.plot(lstm1, color='blue', label='LSTM', linestyle='--')
plt.plot(gru1, color='green', label='GRU', linestyle='--')
plt.plot(actual1, color='black', label='Actual')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Testing period 1')
plt.show()
