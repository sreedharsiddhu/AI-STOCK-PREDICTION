import logging  # Import logging library for logging messages
import requests  # Import requests library for making HTTP requests
import os  # Import os library for interacting with the operating system

from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware for handling cross-origin requests
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler from scikit-learn to scale data
from tensorflow.keras.models import Sequential, load_model  # Import Keras' Sequential model and load_model function
from tensorflow.keras.layers import LSTM, Dense, Dropout  # Import layers for the LSTM model
from fastapi import FastAPI, HTTPException  # Import FastAPI framework and HTTPException for error handling
from pydantic import BaseModel  # Import BaseModel for request validation
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical computations
from datetime import datetime  # Import datetime for date/time manipulation
from fastapi import FastAPI, Request  # Import FastAPI for building the API and Request to access request details




# Setup Logging - Configure the logging to display info-level messages with timestamps
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create FastAPI app instance to define API routes and handle requests
app = FastAPI()

# CORS Middleware - Adds CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin, change to specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)





# Request Model - Define the model for incoming requests with the ticker symbol
class PredictionRequest(BaseModel):
    ticker: str  # A string representing the stock ticker






# Alpha Vantage API Key - Fetch the API key for Alpha Vantage, or use a default one if not found
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "UX1FIZULX8QXBVVM")




# Fetch Stock Data - Function to fetch stock data from Alpha Vantage API
def fetch_stock_data(ticker: str):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={API_KEY}&outputsize=full"  # Build URL for API request
    
    response = requests.get(url)  # Send GET request to Alpha Vantage API
    
    if response.status_code != 200:  # Check if the request was successful
        logging.error(f"Failed to fetch stock data: {response.text}")  # Log an error if failed
        return None  # Return None if request failed

    data = response.json().get("Time Series (Daily)", {})  # Parse the response and extract daily stock data
    
    if not data:  # Check if no data is found for the ticker
        logging.warning(f"No data found for {ticker}")  # Log a warning if no data is found
        return None  # Return None if no data is found

    df = pd.DataFrame(data).T  # Convert the data to a pandas DataFrame and transpose it
    df = df.rename(columns={  # Rename columns for clarity
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume"
    }).astype(float)  # Convert the data to float type
    
    df.index = pd.to_datetime(df.index)  # Convert the index to datetime
    df = df.sort_index()  # Sort the DataFrame by date
    
    return df  # Return the processed DataFrame





# Preprocess Data for LSTM - Function to preprocess stock data for the LSTM model
def preprocess_data(df, sequence_length=30):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize MinMaxScaler to normalize data between 0 and 1
    close_prices = df[['close']].values  # Extract close prices from the DataFrame
    scaled_data = scaler.fit_transform(close_prices)  # Scale the close prices

    X, y = [], []  # Initialize empty lists for the input (X) and target (y) data
    for i in range(sequence_length, len(scaled_data)):  # Loop through the data to create sequences
        X.append(scaled_data[i-sequence_length:i, 0])  # Append the sequence to X
        y.append(scaled_data[i, 0])  # Append the target price to y

    X, y = np.array(X), np.array(y)  # Convert X and y to NumPy arrays
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape X to be compatible with LSTM input

    return X, y, scaler  # Return the preprocessed data and scaler




# Build LSTM Model - Function to build the LSTM model
def build_lstm(sequence_length=30):
    model = Sequential([  # Initialize a Sequential model
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),  # First LSTM layer with 50 units
        Dropout(0.2),  # Dropout layer for regularization
        LSTM(50, return_sequences=False),  # Second LSTM layer with 50 units
        Dropout(0.2),  # Dropout layer for regularization
        Dense(25, activation='relu'),  # Dense layer with 25 units and ReLU activation
        Dense(1)  # Output layer with 1 unit (the predicted price)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model using Adam optimizer and MSE loss function
    return model  # Return the built model

# Load or Train Model - Function to load an existing model or train a new one
def get_or_train_model(X_train, y_train, sequence_length=30):
    model_path = "stock_model.h5"  # Path to save/load the model
    if os.path.exists(model_path):  # Check if the model already exists
        logging.info("Loading existing model...")  # Log the loading of the existing model
        model = load_model(model_path)  # Load the pre-trained model
    else:
        logging.info("Training new model...")  # Log the training of a new model
        model = build_lstm(sequence_length)  # Build a new LSTM model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)  # Train the model with the data
        model.save(model_path)  # Save the trained model to the file
    return model  # Return the model (loaded or trained)




# Root API Endpoint - Serve Frontend (Home page)
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("stock.html", {"request": request})  # Return the index.html template





# Stock Price Prediction API - Endpoint to predict stock prices
@app.get("/predict/{ticker}")
def predict_stock_price(ticker: str):
    logging.info(f"Fetching stock data for {ticker}")  # Log the start of fetching stock data
    
    df = fetch_stock_data(ticker)  # Fetch the stock data
    if df is None:  # If no data is found, raise an HTTP 404 error
        raise HTTPException(status_code=404, detail="No data found for the given ticker")

    X, y, scaler = preprocess_data(df)  # Preprocess the data for LSTM
    train_size = int(len(X) * 0.8)  # 80% of the data will be used for training
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]  # Split the data into training and testing sets

    model = get_or_train_model(X_train, y_train)  # Load or train the model with the training data

    predictions = model.predict(X_test)  # Make predictions using the test data
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))  # Inverse transform the predictions to original scale
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))  # Inverse transform the actual test values to original scale

    return {  # Return the predicted and actual stock prices
        "ticker": ticker,
        "actual_prices": y_test_actual.flatten().tolist(),  # Flatten the array and convert to list
        "predicted_prices": predictions.flatten().tolist(),  # Flatten the array and convert to list
    }
