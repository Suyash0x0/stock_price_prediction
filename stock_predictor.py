
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Download stock data based on a date range
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

# Prepare data for the LSTM model
def prepare_data(data, window_size):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    
    X_train, y_train = [], []
    for i in range(window_size, len(scaled_data)):
        X_train.append(scaled_data[i-window_size:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, scaler

# Build LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and predict
def train_and_predict(ticker, start_date, end_date, window_size=60, epochs=5, batch_size=32):
    data = download_data(ticker, start_date, end_date)
    X_train, y_train, scaler = prepare_data(data, window_size)
    
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    # Predicting on training data
    predicted_stock_price = model.predict(X_train)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    
    # Plotting actual vs predicted
    plt.figure(figsize=(10,6))
    plt.plot(data.values, color='blue', label='Actual Stock Price')
    plt.plot(range(window_size, len(predicted_stock_price) + window_size), predicted_stock_price, color='red', label='Predicted Stock Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig('results/predicted_vs_actual.png')
    plt.show()

if __name__ == "__main__":
    # User input for stock ticker, start date, and end date
    ticker = input("Enter stock ticker symbol (e.g., AAPL for Apple): ")
    start_date = input("Enter the start date for historical data (YYYY-MM-DD): ")
    end_date = input("Enter the end date for historical data (YYYY-MM-DD): ")
    
    train_and_predict(ticker, start_date, end_date)
