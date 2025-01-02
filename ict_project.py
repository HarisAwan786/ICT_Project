import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load Bitcoin data (replace with your actual data file)
data = pd.read_csv('C:/Users/haris/OneDrive/Desktop/ICT PROJECT/bitcoin_project.csv')  # Replace with your CSV path

# Data Preparation
data['DATE'] = pd.to_datetime(data['DATE'])  # Ensure 'DATE' is a datetime object
data['SMA_7'] = data['OPEN'].rolling(window=7).mean()  # 7-day Simple Moving Average
data['SMA_14'] = data['OPEN'].rolling(window=14).mean()  # 14-day Simple Moving Average
data['Momentum'] = data['OPEN'] - data['OPEN'].shift(7)  # Momentum
data.dropna(inplace=True)  # Drop rows with NaN values

# Features and Target
features = ['OPEN', 'SMA_7', 'SMA_14', 'Momentum']
scaler = MinMaxScaler()
X = scaler.fit_transform(data[features])  # Scale features
y = data['OPEN'].values

# Train-test split
train_size = int(len(data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict future prices based on historical data
future_days = 7
predicted_prices = []

# Predict for the next 7 days
for i in range(future_days):
    X_last = X[-(future_days - i)].reshape(1, -1)  # Reshape last known data point for prediction
    prediction = model.predict(X_last)[0]  # Predict the price
    predicted_prices.append(prediction)

# Function to display actual graph
def show_graph():
    plt.figure(figsize=(10, 6))
    plt.plot(data['DATE'], data['OPEN'], label="Actual Prices", color="blue", marker='o')
    plt.title("Bitcoin Price (Dec 1 - Jan 1)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.xticks(data['DATE'], rotation=45)  # Display all dates on x-axis
    plt.xticks(pd.date_range(start=data['DATE'].min(), end=data['DATE'].max(), freq='D'), rotation=45)  # Ensure all dates from Dec 1 to Jan 1 are shown
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent cutoff
    plt.show()

# Function to display predicted graph
def show_predicted_graph():
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, future_days + 1), predicted_prices, marker='o', label="Predicted Prices", color="green")
    plt.title("Predicted Bitcoin Prices for Next 7 Days")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to display accuracy
def show_accuracy():
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    accuracy = 100 - (mae / np.mean(y_test) * 100)
    accuracy_label.config(text=f"Model Accuracy: {accuracy:.2f}%")

# Tkinter Setup
root = tk.Tk()
root.title("Bitcoin Price Prediction")
root.geometry("800x600")
root.configure(bg="#1b1b1b")  # Dark background

# Font Styles
bold_font = font.Font(family="Helvetica", size=16, weight="bold")
regular_font = font.Font(family="Helvetica", size=14)

# Title Label
title_label = tk.Label(root, text="Bitcoin Price Prediction", font=bold_font, fg="#FFFFFF", bg="#1b1b1b")
title_label.pack(pady=20)

# Prediction Display Frame
prediction_frame = tk.Frame(root, bg="#282828", padx=20, pady=10)
prediction_frame.pack(pady=20)

# Display Predictions
tk.Label(prediction_frame, text="Predicted Prices for the Next Week:", font=bold_font, fg="#00FFAA", bg="#282828").pack(anchor="w")
days_of_week = ["Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"]
for i, price in enumerate(predicted_prices):
    day_text = f"{days_of_week[i]}: ${price:.2f}"
    tk.Label(prediction_frame, text=day_text, font=regular_font, fg="#FFFFFF", bg="#282828").pack(anchor="w")

# Accuracy Display
accuracy_label = tk.Label(root, text="", font=bold_font, fg="#FFFFFF", bg="#1b1b1b")
accuracy_label.pack(pady=10)

# Button Container
button_frame = tk.Frame(root, bg="#1b1b1b")
button_frame.pack(pady=20)

# Graph Buttons
graph_button = tk.Button(button_frame, text="Show Graph", font=regular_font, bg="#00AAFF", fg="white", command=show_graph, relief="flat", padx=10, pady=5)
graph_button.grid(row=0, column=0, padx=10)

predicted_graph_button = tk.Button(button_frame, text="Show Predicted Graph", font=regular_font, bg="#00AAFF", fg="white", command=show_predicted_graph, relief="flat", padx=10, pady=5)
predicted_graph_button.grid(row=0, column=1, padx=10)

accuracy_button = tk.Button(button_frame, text="Show Accuracy", font=regular_font, bg="#00AAFF", fg="white", command=show_accuracy, relief="flat", padx=10, pady=5)
accuracy_button.grid(row=0, column=2, padx=10)

# Close Button
close_button = tk.Button(button_frame, text="Close", font=regular_font, bg="#FF5733", fg="white", command=root.destroy, relief="flat", padx=10, pady=5)
close_button.grid(row=0, column=3, padx=10)

# Run Tkinter Loop
root.mainloop()

