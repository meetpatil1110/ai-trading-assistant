from fastapi import FastAPI, HTTPException
import numpy as np
import pickle
import os
from pydantic import BaseModel
from tensorflow.keras.models import load_model

app = FastAPI()

# Load model and scaler on startup
try:
    model = load_model("lstm_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    raise Exception(f"Error loading model or scaler: {str(e)}")


class InputData(BaseModel):
    prices: list


@app.get("/")
def read_root():
    return {"message": "LSTM Prediction API is running"}


@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert prices to numpy array
        prices = np.array(data.prices).reshape(-1, 1)
        
        # Scale the input data
        scaled_prices = scaler.transform(prices)
        
        # Reshape for LSTM (samples, timesteps, features)
        X = scaled_prices.reshape(1, -1, 1)
        
        # Make prediction
        prediction = model.predict(X, verbose=0)
        
        # Inverse transform to get actual price
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        
        return {"predicted_price": float(predicted_price)}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
