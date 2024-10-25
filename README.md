
# Stock Prediction using LSTM

This project predicts stock prices using a Long Short-Term Memory (LSTM) model, a type of neural network suited for time-series data like stock prices.

## Features
- Fetches historical stock data using the `yfinance` API.
- Trains an LSTM model to predict future stock prices.
- Visualizes the actual and predicted stock prices.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/Suyash0x0/stock_price_prediction.git
    cd stock_price_prediction
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the project:
    ```bash
    python stock_predictor.py
    ```

## Example Output

A plot of the actual vs predicted stock prices is saved in the `results/` folder as `predicted_vs_actual.png`.

## License
This project is licensed under the MIT License.
