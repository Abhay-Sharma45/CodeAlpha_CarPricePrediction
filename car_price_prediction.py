"""
CodeAlpha - Task 3: Car Price Prediction with Machine Learning
Trains a regression model to predict a used car's selling price.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("data/car data.csv")

print("Shape:", df.shape)
print(df.head())
print(df.isnull().sum())

# ---------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
sns.pairplot(df, hue="Fuel_Type")
plt.savefig("eda_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 3. Feature engineering & preprocessing
# ---------------------------------------------------------
df["Car_Age"] = 2026 - df["Year"]
df_model = df.drop(columns=["Car_Name", "Year"])

le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_trans = LabelEncoder()
df_model["Fuel_Type"] = le_fuel.fit_transform(df_model["Fuel_Type"])
df_model["Selling_type"] = le_seller.fit_transform(df_model["Selling_type"])
df_model["Transmission"] = le_trans.fit_transform(df_model["Transmission"])

X = df_model.drop(columns=["Selling_Price"])
y = df_model["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nR2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color="darkorange")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price")
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 6. Feature importance
# ---------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

plt.figure(figsize=(6, 4))
importances.sort_values().plot(kind="barh", color="mediumseagreen")
plt.title("Feature Importance")
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nDone. Plots saved: eda_pairplot.png, correlation_heatmap.png, actual_vs_predicted.png, feature_importance.png")
