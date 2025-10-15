# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model at startup
model = joblib.load("sales_prediction_model.pkl")

app = FastAPI(
    title="Nike Future Sales Predictor",
    description="Predict next-period sales using product features and past sales.",
    version="1.0"
)

class SalesInput(BaseModel):
    sale_price: float
    rating: float
    review_posted: int
    sold_products: int

@app.get("/")
def home():
    return {"message": "Nike Sales Prediction API is running!"}

@app.post("/predict")
def predict(input_data: SalesInput):
    # Prepare input for model
    features = np.array([[
        input_data.sale_price,
        input_data.rating,
        input_data.review_posted,
        input_data.sold_products
    ]])
    
    # Predict
    prediction = model.predict(features)
    
    # Return integer prediction
    return {
        "predicted_future_sales": int(round(prediction[0]))
    }