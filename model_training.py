# model_training.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load preprocessed data
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=1)
model.fit(X_train, Y_train, batch_size=128, epochs=1, validation_data=(X_test, Y_test), callbacks=[early_stopping])

# Save the model
model.save('lstm_model.h5')

# Evaluate the model
loss = model.evaluate(X_test, Y_test)
print('Test Loss:', loss)