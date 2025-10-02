# Stock Price Forecasting Web App

A Flask-based web application that predicts stock prices using **ARIMA**, **Prophet**, and **LSTM** models, with interactive visualization and accuracy metrics.

---

## Features

- Predicts next 30 days’ stock prices for selected tickers.
- Compares three different forecasting models: ARIMA, Prophet, and LSTM.
- Interactive Plotly chart with zoom, hover, and range slider.
- Highlights model accuracy using RMSE.
- Dark-themed professional UI for clear visualization.

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py

# 5. Open in browser
http://127.0.0.1:5000

```

## Usage

1. Select a stock ticker from the dropdown (AAPL, MSFT, GOOGL).
2. Click **Submit**.
3. View the **forecast chart** and **RMSE results**.
4. Hover over the chart to see exact predicted and actual prices.


## How it works

1. **Data Collection**: Fetches historical stock data from Yahoo Finance using `yfinance`.
2. **ARIMA Model**: Uses `statsmodels` for traditional time series forecasting.
3. **Prophet Model**: Uses Facebook Prophet for trend and seasonality analysis.
4. **LSTM Model**: Deep learning model trained on historical prices for sequence prediction.
5. **Forecast Generation**: Each model predicts the next 30 days’ stock prices.
6. **Evaluation**: Compares RMSE of each model to highlight the most accurate prediction.
7. **Visualization**: Plotly renders an interactive, dark-themed chart comparing all models with actual prices.


