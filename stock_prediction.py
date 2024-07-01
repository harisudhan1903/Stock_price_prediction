# Install necessary packages
pip install yfinance
pip install tensorflow
pip install scikit-learn

# Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Dataset from wipro
df = yf.download('WIPRO.NS')
df.to_csv('wipro_stock_data.csv')

print(df.head())
print(df.tail())

# Plot the historical closing prices
plt.figure(figsize=(16,8))
plt.title('Historical Price')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

# Filter the close price
data = df.filter(['Close'])
df_array = np.array(data).reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_df = scaler.fit_transform(df_array)

# Define the training data length
training_data_len = math.ceil(len(scaled_df) * 0.8)

# Create the training data
train_data = scaled_df[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 60:
        print(x_train)
        print(y_train)
        print()

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=40, epochs=20)

# Create the testing data
test_data = scaled_df[training_data_len - 60:, :]
x_test = []
y_test = df_array[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Accuracy metrics calculation
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R2 Score: {r2}')

# Plot the data
train = data[:training_data_len]
val = data[training_data_len:]
val['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train['Close'])
plt.plot(val[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

plt.figure(figsize=(16, 8))
plt.title('Actual vs Predicted Prices')
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(predictions, color='red', label='Predicted Prices')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.legend()
plt.show()

plt.figure(figsize=(16, 8))
plt.title('Prediction Error')
plt.plot(y_test - predictions, color='purple', label='Prediction Error')
plt.xlabel('Days')
plt.ylabel('Error')
plt.legend()
plt.show()