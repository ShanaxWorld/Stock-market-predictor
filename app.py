# app.py
from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import warnings

warnings.filterwarnings("ignore")  # Hide warnings for clean logs

app = Flask(__name__)

# ----------------------
# Prepare LSTM data
# ----------------------
def prepare_lstm_data(series, window=60):
    X, y = [], []
    for i in range(len(series)-window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)

# ----------------------
# Forecast functions
# ----------------------
def arima_forecast(model, steps=30):
    if model is None:
        return [np.nan]*steps
    history = model.data.endog.tolist()
    predictions = []
    for _ in range(steps):
        forecast = model.forecast()
        predictions.append(forecast.iloc[0])
        history.append(forecast.iloc[0])
    return predictions

def prophet_forecast(model, df_prophet, steps=30):
    if model is None or df_prophet is None:
        return [np.nan]*steps
    future = df_prophet[['ds']].iloc[-steps:]
    forecast = model.predict(future)
    return forecast['yhat'].values

def lstm_forecast(model, series, window=60, steps=30):
    if model is None or len(series) < window:
        return [np.nan]*steps
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_series = scaler.fit_transform(series.reshape(-1,1))
    last_window = scaled_series[-window:].reshape(1, window,1)
    preds = []
    for _ in range(steps):
        pred = model.predict(last_window, verbose=0)[0,0]
        preds.append(pred)
        # update last_window
        last_window = np.append(last_window[:,1:,:], [[[pred]]], axis=1)
    return scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

# ----------------------
# Pre-train models for tickers
# ----------------------
tickers = ["AAPL","MSFT","GOOGL"]
models_dict = {}

for ticker in tickers:
    try:
        df = yf.download(ticker, start="2018-01-01", end="2025-01-01", threads=False)[['Close']]
        df.dropna(inplace=True)
        if df.empty or len(df) < 20:
            print(f"Skipping {ticker}, not enough data")
            continue
        series = df['Close']

        # ARIMA
        try:
            arima_model = ARIMA(series, order=(5,1,0)).fit()
        except Exception as e:
            print(f"ARIMA failed for {ticker}: {e}")
            arima_model = None

        # Prophet
        df_prophet = df.reset_index()
        df_prophet.columns = ['ds','y']
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet.dropna(inplace=True)
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        try:
            prophet_model = Prophet(daily_seasonality=True)
            prophet_model.fit(df_prophet)
        except Exception as e:
            print(f"Prophet failed for {ticker}: {e}")
            prophet_model = None

        # LSTM
        lstm_series = series.values
        window = 60
        if len(lstm_series) > window:
            X, y = prepare_lstm_data(lstm_series, window)
            X = X.reshape((X.shape[0], X.shape[1],1))
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(window,1)),
                LSTM(50),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X, y, epochs=5, verbose=0)
        else:
            lstm_model = None

        # Store models and data
        models_dict[ticker] = {
            "df": df,
            "df_prophet": df_prophet,
            "arima": arima_model,
            "prophet": prophet_model,
            "lstm": lstm_model
        }
        print(f"{ticker} models trained successfully.")

    except Exception as e:
        print(f"Download or training failed for {ticker}: {e}")

# ----------------------
# Flask Routes
# ----------------------
@app.route("/", methods=["GET","POST"])
def index():
    chart_div = ""
    accuracy_info = ""
    if request.method == "POST":
        ticker = request.form.get("ticker")
        if ticker not in models_dict:
            return "Ticker not supported!"

        data = models_dict[ticker]
        df = data["df"]
        steps = 30

        # Forecasts
        arima_pred = arima_forecast(data["arima"], steps)
        prophet_pred = prophet_forecast(data["prophet"], data["df_prophet"], steps)
        lstm_pred = lstm_forecast(data["lstm"], df['Close'].values, window, steps)

        # Metrics: simple RMSE comparison (skip NaN)
        actual = df['Close'].values[-steps:]
        rmse_dict = {}
        if data["arima"]: rmse_dict["ARIMA"] = np.sqrt(np.mean((np.array(arima_pred)-actual)**2))
        if data["prophet"]: rmse_dict["Prophet"] = np.sqrt(np.mean((np.array(prophet_pred)-actual)**2))
        if data["lstm"]: rmse_dict["LSTM"] = np.sqrt(np.mean((np.array(lstm_pred)-actual)**2))
        best_model = min(rmse_dict, key=rmse_dict.get) if rmse_dict else "None"
        accuracy_info = ", ".join([f"{k}: {v:.2f}" for k,v in rmse_dict.items()])
        accuracy_info = f"RMSE -> {accuracy_info}. Best: {best_model}"

        # Plot with Plotly
        # Plot with dark theme
        trace1 = go.Scatter(
            x=df.index[-steps:], y=arima_pred, mode='lines', name="ARIMA",
            line=dict(color='red', width=3),
            hovertemplate='ARIMA: %{y:.2f}<extra></extra>'
        )
        trace2 = go.Scatter(
            x=df.index[-steps:], y=prophet_pred, mode='lines', name="Prophet",
            line=dict(color='green', width=3),
            hovertemplate='Prophet: %{y:.2f}<extra></extra>'
        )
        trace3 = go.Scatter(
            x=df.index[-steps:], y=lstm_pred, mode='lines', name="LSTM",
            line=dict(color='blue', width=3),
            hovertemplate='LSTM: %{y:.2f}<extra></extra>'
        )
        trace4 = go.Scatter(
            x=df.index[-steps:], y=actual, mode='lines', name="Actual",
            line=dict(color='white', width=2, dash='dash'),
            hovertemplate='Actual: %{y:.2f}<extra></extra>'
        )

        layout = go.Layout(
            title=f"{ticker} Stock Forecast (Next {steps} Days)",
            xaxis=dict(title="Date", color='white'),
            yaxis=dict(title="Price", color='white'),
            plot_bgcolor='#1e1e2f',
            paper_bgcolor='#1e1e2f',
            font=dict(color='white'),
            legend=dict(bgcolor='#2b2b3f', bordercolor='white', borderwidth=1)
        )

        fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
        chart_div = fig.to_html(full_html=False)

    return render_template("index.html", tickers=list(models_dict.keys()), chart_div=chart_div, accuracy_info=accuracy_info)

if __name__ == "__main__":
    app.run(debug=True)









