import logging  # For logging messages
import os  # For accessing environment variables and file paths
from datetime import datetime  # For handling dates and times

import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation and analysis
import requests  # For making HTTP requests to fetch stock data
from fastapi import FastAPI, Request, HTTPException  # For creating the API and handling errors
from fastapi.middleware.cors import CORSMiddleware  # For enabling CORS
from fastapi.responses import HTMLResponse  # For returning HTML responses
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel  # For data validation and settings management
from sklearn.preprocessing import MinMaxScaler  # For scaling data
from tensorflow.keras.layers import LSTM, Dense, Dropout  # For building the LSTM model
from tensorflow.keras.models import Sequential, load_model  # For creating the neural network and loading models
from keras_tuner.tuners import RandomSearch  # For hyperparameter tuning

# Configure logging to track the application's behavior
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create a FastAPI instance
app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(os.getcwd(), "templates"))



# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,  # Allow credentials (e.g., cookies)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Alpha Vantage API Key (move to environment variable for security)
# API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "UX1FIZULX8QXBVVM")"DTXPY0JRPPT66VLZ"
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "UX1FIZULX8QXBVVM")

def fetch_stock_data(ticker: str):
    """
    Fetch stock data from Alpha Vantage API.
    :param ticker: Stock symbol (e.g., "AAPL").
    :return: DataFrame containing stock data or None if fetching fails.
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={API_KEY}&outputsize=full"
    
    # Make an HTTP GET request to the API
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch stock data: {response.text}")
        return None

    # Parse the JSON response
    json_data = response.json()

    # Check for API errors
    if "Error Message" in json_data:
        logging.error(f"API Error: {json_data['Error Message']}")
        return None
    if "Information" in json_data:
        logging.warning(f"API Information: {json_data['Information']}")
        return None

    # Extract the "Time Series (Daily)" data from the response
    data = json_data.get("Time Series (Daily)", {})
    
    # Check if data is empty
    if not data:
        logging.warning(f"No data found for {ticker}")
        return None

    # Convert the data into a DataFrame
    df = pd.DataFrame(data).T
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume"
    }).astype(float)

    # Convert the index to datetime and sort by date
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    return df
# Preprocess data
def preprocess_data(df, sequence_length=30):
    """
    Preprocess stock data for LSTM model.
    :param df: DataFrame containing stock data.
    :param sequence_length: Length of the input sequence for LSTM.
    :return: Scaled features (X), target (y), and the scaler object.
    """
    # Scale the "close" prices to a range of 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = df[['close']].values
    scaled_data = scaler.fit_transform(close_prices)

    # Create sequences of data for LSTM
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    # Convert lists to numpy arrays and reshape X for LSTM input
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Hyperparameter tuning
def build_lstm_model(hp):
    """
    Build an LSTM model with hyperparameter tuning.
    :param hp: Hyperparameter object from Keras Tuner.
    :return: Compiled LSTM model.
    """
    model = Sequential()

    # Tune the number of LSTM units
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        return_sequences=True,
        input_shape=(30, 1)  # Input shape is fixed (sequence_length=30, features=1)
    ))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

    # Tune the number of LSTM layers
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(LSTM(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            return_sequences=(i < 2)  # Only return sequences for intermediate layers
        ))
        model.add(Dropout(hp.Float(f'dropout_{i+1}', min_value=0.1, max_value=0.5, step=0.1)))

    # Add a dense layer with ReLU activation
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=128, step=16), activation='relu'))

    # Add the output layer
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Train or load model
def get_or_train_model(X_train, y_train, epochs=20):  # Added epochs parameter
    """
    Train or load an LSTM model with hyperparameter tuning.
    :param X_train: Training features.
    :param y_train: Training targets.
    :param epochs: Number of epochs to train the model.
    :return: Trained or loaded LSTM model.
    """
    model_path = "stock_model.h5"

    # Check if a pre-trained model exists
    if os.path.exists(model_path):
        logging.info("Loading existing model...")
        model = load_model(model_path)
    else:
        logging.info("Training new model with hyperparameter tuning...")

        # Initialize Keras Tuner
        tuner = RandomSearch(
            build_lstm_model,
            objective='val_loss',
            max_trials=5,  # Number of hyperparameter combinations to try
            executions_per_trial=2,  # Number of models to train per trial
            directory='tuner_results',  # Directory to save tuning results
            project_name='stock_prediction'
        )

        # Perform hyperparameter search
        tuner.search(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=1)

        # Get the best model
        best_model = tuner.get_best_models(num_models=1)[0]

        # Save the best model
        best_model.save(model_path)
        model = best_model

    return model

# Root route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Render the home page using Jinja2.
    """
    return templates.TemplateResponse("stock.html", {"request": request})

# Prediction route
@app.get("/predict/{ticker}")
def predict_stock_price(ticker: str):
    """
    Predict stock prices for a given ticker.
    :param ticker: Stock symbol (e.g., "AAPL").
    :return: JSON response containing actual and predicted prices.
    """
    logging.info(f"Fetching stock data for {ticker}")
    
    # Fetch stock data
    df = fetch_stock_data(ticker)
    if 'Information' in df:
        return {"Issue": df['Information']}
    if df is None:
        raise HTTPException(status_code=404, detail="No data found for the given ticker")

    # Preprocess data
    X, y, scaler = preprocess_data(df)
    train_size = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # Load or train the model
    model = get_or_train_model(X_train, y_train, epochs=20)  # Train for 20 epochs

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Add dates to the response
    dates = df.index[train_size + 30:].strftime('%Y-%m-%d').tolist()  # Adjusted for sequence length

    return {
        "ticker": ticker,
        "dates": dates,  # Include dates in the response
        "actual_prices": y_test_actual.flatten().tolist(),
        "predicted_prices": predictions.flatten().tolist(),
    }

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)