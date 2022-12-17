#!/usr/bin/env python3


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from pathlib import Path


def write_to_csv(index=False):
    outpath = Path("./results.csv")
    results.to_csv(outpath, index=index)

def plot_values():
    # Plot the predicted values against the actual values
    figure, axis = plt.subplots(2, 2)
    figure.set_size_inches(20, 18.5)
    axis[0,0].plot(results['percent_diff'], label="Percent Diff")
    axis[0,0].legend()
    axis[0,1].plot(y_test, label="Actual")
    axis[0,1].plot(predictions, label="Predicted")
    axis[0,1].legend()
    # TODO - check on the order here
    axis[1,0].bar(["True", "False"], correct_counts, label="Correct Direction")
    axis[1,0].legend()
    plt.savefig("plt.png", dpi=100)


symbol = yf.Ticker("SPY")
df = symbol.history(period="max", interval="1mo")[["Close"]]
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()
df.reset_index(inplace=True)

# Convert the Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Set the Date column as the index of the DataFrame
df.set_index("Date", inplace=True)

# Split the data into a training set and a test set
train_data, test_data = train_test_split(df, train_size=0.8, shuffle=False)
print(f"test_data {test_data}")

# Normalize the data using the MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# Convert the training and test sets to NumPy arrays
X_train = np.array(train_data[:-1])
y_train = np.array(train_data[1:])
X_test = np.array(test_data[:-1])
y_test = np.array(test_data[1:])

# Reshape the data for use with an LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(loss="mean_absolute_error", optimizer="adam")

# Fit the model to the training data
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1)

# Evaluate the model on the test data
score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {score:.3f}")

# Make predictions on new data
predictions = model.predict(X_test)

y_test = pd.DataFrame(y_test)
predictions = pd.DataFrame(predictions)

# make a results df to store analysis
results = pd.DataFrame()
results['percent_diff'] = (y_test - predictions) / y_test
results['true_direction'] = y_test - y_test.shift(1)
results['pred_direction'] = predictions - predictions.shift(1)

# check if matching sign
results['correct'] = results['true_direction'] * results['pred_direction'] >= 0
correct_counts = results.correct.value_counts()
print(type(correct_counts))
print(correct_counts)

plot_values()
write_to_csv()
