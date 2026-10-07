import pandas as pd
import numpy as np

df = pd.read_csv("bike_sales_numeric_sp.csv")
print("Total rows:", len(df))
print("Models:", df["Bike_Model"].unique().tolist())
for model in df["Bike_Model"].unique():
    mdf = df[df["Bike_Model"] == model]
    print(f"\nModel: {model}")
    print(f"Price: min={mdf['Price'].min():.2f}, mean={mdf['Price'].mean():.2f}, max={mdf['Price'].max():.2f}")
    print(f"Quantity: min={mdf['Quantity'].min()}, mean={mdf['Quantity'].mean():.2f}, max={mdf['Quantity'].max()}")
    print(f"Corr(Price, Quantity): {mdf['Price'].corr(mdf['Quantity']):.4f}")
    print(f"Corr(Price, Total_Sales): {mdf['Price'].corr(mdf['Total_Sales']):.4f}")
