import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import os

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL","GOOG", "MSFT", "GME")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1,4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data ... done!")

st.subheader("Raw data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date":"ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Prophet forecast data")
st.write(forecast.tail())

st.write("forecast data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)



#Linear model
st.subheader("Linear model")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import plotly.graph_objs as go
import numpy as np

layout= go.Layout(
    title="Stock prices",
    xaxis=dict(
        title="Date",
        titlefont=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
        )
    ),
    yaxis=dict(
            title="Price",
            titlefont=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
            )
        )
)

lm_data = [{'x': data.index, 'y': data['Close']}]
plot = go.Figure(data=lm_data, layout=layout)

X = np.array(data.index).reshape(-1,1)
Y = data['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, Y_train)

trace0= go.Scatter(
    x = X_train.T[0],
    y = Y_train,
    mode = 'markers',
    name = 'Actual'
)

trace1= go.Scatter(
    x = X_train.T[0],
    y = lm.predict(X_train).T,
    mode = 'lines',
    name = 'Predicted'
)

predict_data = [trace0, trace1]
layout.xaxis.title.text = "Day"
plot2=go.Figure(data=predict_data, layout=layout)
plot2

#Calculate scores for model evaluation
st.write('Metrics - Train - r2_score')
r2_score_train = r2_score(Y_train, lm.predict(X_train))
r2_score_train

st.write('Metrics - Test - r2_score')
r2_score_test = r2_score(Y_test, lm.predict(X_test))
r2_score_test

st.write('Metrics - Train - MSE')
mse_train = mse(Y_train, lm.predict(X_train))
mse_train

st.write('Metrics - Test - MSE')
mse_test = mse(Y_test, lm.predict(X_test))
mse_test


#Handle data
data.index = pd.to_datetime(data["Date"])
del data["Date"]
data.plot.line(y="Close", use_index=True)
data["Tomorrow"] = data["Close"].shift(-1)
data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
data = data.loc["2013-01-01":].copy()

st.subheader("RandomForestClassifier")
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = data.iloc[:-100]
test = data.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

st.subheader("precision_score")
st.write("1 - Up")
st.write("0 - Down")

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

precision_score(test["Target"], preds)

combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(dataInput, model, predictors, start=100, step=250):
    all_predictions = []

    for i in range(start, dataInput.shape[0], step):
        train = dataInput.iloc[0:i].copy()
        test = dataInput.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)

predictions = backtest(data, model, predictors)

predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])

precision_score(predictions["Target"], predictions["Predictions"])

predictions["Target"].value_counts() / predictions.shape[0]

horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = data.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    data[ratio_column] = data["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]

    new_predictors+= [ratio_column, trend_column]


data = data.dropna(subset=data.columns[data.columns != "Tomorrow"])
data

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    #If it goes up
    preds[preds >=.6] = 1
    #If it goes down
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

predictions = backtest(data, model, new_predictors)

predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])

predictions["Target"].value_counts() / predictions.shape[0]
predictions
def plot_predict_data(input):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=input.index, y=input['Predictions'], name='Predictions'))
    fig.layout.update(title_text="Time Series Predict Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_predict_data(predictions)
