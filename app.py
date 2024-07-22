import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from pyngrok import ngrok

# Load the trained model
model = load_model('lstm_model.h5')

# Load the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.scale_ = np.load('scaler.npy')

# Streamlit app
st.title('Stock Price Prediction')

# Select the stock for prediction
ticker = st.text_input('Enter stock ticker')

# Load stock data for the selected ticker
if ticker:
    data = pd.read_csv('VAR.csv', parse_dates=True, index_col='Date')
    data = data[data['Ticker'] == ticker]

    if not data.empty:
        st.write(f'Stock data for {ticker}')
        st.line_chart(data['Close'])

        # Prepare data for prediction
        scaled_data = scaler.transform(data[['Close']])
        look_back = 60
        X = []
        for i in range(len(scaled_data) - look_back - 1):
            X.append(scaled_data[i:(i + look_back), 0])
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Make predictions
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)

        # Show predictions
        prediction_dates = data.index[look_back + 1:]
        prediction_df = pd.DataFrame(data={'Date': prediction_dates, 'Predicted Close': predictions.flatten()})
        st.write('Predicted Close Prices')
        st.line_chart(prediction_df.set_index('Date'))
    else:
        st.write(f'No data available for {ticker}')

# Create a public URL for Streamlit app using pyngrok
public_url = ngrok.connect(port='8501')
print('Public URL:', public_url)
