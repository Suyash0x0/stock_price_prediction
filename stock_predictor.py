import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to prepare the data for LSTM
def prepare_data(data, window_size):
    print("Data shape:", data.shape)  # Check the shape of the data

    # Ensure there are enough data points
    if len(data) < window_size:
        raise ValueError(f"Not enough data to create training sequences. "
                         f"Expected at least {window_size} data points, but got {len(data)}.")

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close']])  # Use only 'Close' prices

    # Create the training sequences
    X_train, y_train = [], []
    for i in range(window_size, len(data_scaled)):
        X_train.append(data_scaled[i-window_size:i, 0])
        y_train.append(data_scaled[i, 0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Reshape for LSTM [samples, time steps, features]
    if X_train.shape[0] == 0:
        raise ValueError("X_train is empty after preparation.")

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, scaler

# Function to create and train the LSTM model
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of stock price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main function to download data, prepare it, and train the model
def train_and_predict(ticker, start_date, end_date, window_size=5):
    data = yf.download(ticker, start=start_date, end=end_date)
    print("Retrieved data:\n", data)

    if data.empty:
        print("No data found for the specified date range.")
        return

    X_train, y_train, scaler = prepare_data(data, window_size)

    model = create_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    print("Model training completed.")

    # Make predictions for the next day
    last_data = data[['Close']].values[-window_size:]  # Get the last 'window_size' data points
    last_data_scaled = scaler.transform(last_data)
    X_test = np.reshape(last_data_scaled, (1, last_data_scaled.shape[0], 1))

    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)  # Inverse transform to get actual price
    
    print(f"Predicted price for {ticker} on the next day: ${predicted_price[0][0]:.2f}")

if __name__ == "__main__":
    # User inputs for the stock prediction
    ticker = input("Enter stock ticker symbol (e.g., AAPL for Apple): ")
    start_date = input("Enter the start date for historical data (YYYY-MM-DD): ")
    end_date = input("Enter the end date for historical data (YYYY-MM-DD): ")
    
    # Train the model and predict
    train_and_predict(ticker, start_date, end_date)
