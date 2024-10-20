

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#using historical apple data and only predicting closing prices for now
data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')
closing_prices = data['Close'].values.reshape(-1, 1)

# using moving avg and volume to increase accuracy of prediction
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
 # drop NaNs created by rolling windows
data = data.dropna()
features = data[['Close', 'MA50', 'MA200', 'Volume']].values


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

#making input sequences for  LSTM
def create_sequences(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


time_step = 60
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

X_train, y_train = create_sequences(train_data, time_step)
X_test, y_test = create_sequences(test_data, time_step)

# input should be in the format of [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))


model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(time_step, X_train.shape[2])))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))


optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_lstm_model_v3.keras', save_best_only=True, monitor='val_loss')


history = model.fit(X_train, y_train, epochs=50, batch_size=1, validation_split=0.1, callbacks=[early_stop, checkpoint])


lstm_predictions = model.predict(X_test)
lstm_predictions = scaler.inverse_transform(np.concatenate([lstm_predictions, np.zeros((len(lstm_predictions), 3))], axis=1))[:, 0]


actual_prices = data['Close'][train_size + time_step : train_size + time_step + len(lstm_predictions)]

rmse = np.sqrt(mean_squared_error(actual_prices, lstm_predictions))
print(f"RMSE: {rmse}")

#visualizations
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index[train_size+time_step:], y=actual_prices, name='Actual Prices', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data.index[train_size+time_step:], y=lstm_predictions.flatten(), name='LSTM Predictions', line=dict(color='red')))
fig.update_layout(title='LSTM Model Predictions vs Actual Prices', xaxis_title='Date', yaxis_title='Stock Price')
fig.show()
