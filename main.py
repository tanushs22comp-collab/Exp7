from fastapi import FastAPI
from pydantic import BaseModel
import joblib


# Load model at startup
model = joblib.load("sales_prediction_model.pkl")


app = FastAPI(
    title="Nike Future Sales Predictor",
    description=(
        "Predict next-period sales using product features and past sales."
    ),
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
def predict(input_ SalesInput):
    features = [[
        input_data.sale_price,
        input_data.rating,
        input_data.review_posted,
        input_data.sold_products
    ]]
    prediction = model.predict(features)
    return {
        "predicted_future_sales": int(round(prediction[0]))
    }
