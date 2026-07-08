# CodeAlpha_CarPricePrediction

**Task 3 — Car Price Prediction with Machine Learning** (CodeAlpha Data Science Internship)

Predicts the selling price of a used car using features like present price,
kilometers driven, fuel type, seller type, transmission, and car age, using
a Random Forest Regressor.

## Project Structure
```
CodeAlpha_CarPricePrediction/
├── data/
│   └── car data.csv
├── car_price_prediction.py
├── requirements.txt
└── README.md
```

## Setup (Mac)
```bash
cd CodeAlpha_CarPricePrediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python car_price_prediction.py
```

## What it does
1. Loads and explores the dataset (pairplot + correlation heatmap).
2. Engineers a `Car_Age` feature and encodes categorical variables
   (Fuel_Type, Selling_type, Transmission).
3. Splits data into train/test sets.
4. Trains a Random Forest Regressor.
5. Evaluates performance using R², MAE, and RMSE, and plots actual vs.
   predicted prices.
6. Reports which features matter most for predicting price.

## Results
Typically achieves an R² score of ~0.95 on the test set — `Present_Price`
is by far the strongest predictor of selling price, followed by car age
and kilometers driven.

## Dataset
[Vehicle dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) — 301 samples, used car listings with price and specs.
