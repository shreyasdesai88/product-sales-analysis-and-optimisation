import pandas as pd
import numpy as np

df = pd.read_csv("bike_sales_numeric_sp.csv")
model = "KTM 390 Adventure"
mdf = df[df["Bike_Model"] == model].copy()

# Look at distribution of prices
print("Price percentiles:")
print(mdf["Price"].quantile([0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]))

# Look at weekly / monthly aggregate volume and price
mdf["Date"] = pd.to_datetime(mdf["Date"])
mdf["Week"] = mdf["Date"].dt.to_period("W")
weekly = mdf.groupby("Week").agg(
    avg_price=("Price", "mean"),
    total_qty=("Quantity", "sum"),
    total_revenue=("Total_Sales", "sum"),
    tx_count=("Sale_ID", "count")
)
print("\nWeekly corr(avg_price, total_qty):", weekly["avg_price"].corr(weekly["total_qty"]))
print("Weekly corr(avg_price, tx_count):", weekly["avg_price"].corr(weekly["tx_count"]))
print("Weekly corr(avg_price, total_revenue):", weekly["avg_price"].corr(weekly["total_revenue"]))

# Look at price bins vs volume
mdf["price_bin"] = pd.qcut(mdf["Price"], q=10)
binned = mdf.groupby("price_bin", observed=False).agg(
    mean_price=("Price", "mean"),
    sales_count=("Sale_ID", "count"),
    total_qty=("Quantity", "sum"),
    mean_qty=("Quantity", "mean"),
    total_revenue=("Total_Sales", "sum")
)
print("\nBinned by price decile:")
print(binned)
