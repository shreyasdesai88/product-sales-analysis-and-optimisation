import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("bike_sales_numeric_sp.csv")
model = "KTM 390 Adventure"
mdf = df[df["Bike_Model"] == model].copy()
mdf["Date"] = pd.to_datetime(mdf["Date"])

# Aggregate by week
weekly = mdf.groupby(pd.Grouper(key="Date", freq="W")).agg(
    avg_price=("Price", "mean"),
    total_qty=("Quantity", "sum"),
    total_sales=("Total_Sales", "sum"),
    count=("Sale_ID", "count")
).dropna()

print("Weekly data points:", len(weekly))
print("Weekly avg price mean:", weekly["avg_price"].mean(), "min:", weekly["avg_price"].min(), "max:", weekly["avg_price"].max())

# Regress total_qty on avg_price
lr_qty = LinearRegression()
lr_qty.fit(weekly[["avg_price"]], weekly["total_qty"])
print("Weekly Qty ~ Price slope:", lr_qty.coef_[0], "intercept:", lr_qty.intercept_)

# Regress count on avg_price
lr_cnt = LinearRegression()
lr_cnt.fit(weekly[["avg_price"]], weekly["count"])
print("Weekly Count ~ Price slope:", lr_cnt.coef_[0], "intercept:", lr_cnt.intercept_)

# Price elasticity at mean
p_mean = weekly["avg_price"].mean()
q_mean = weekly["total_qty"].mean()
ped = -lr_qty.coef_[0] * (p_mean / q_mean)
print("Weekly Price Elasticity of Demand (PED):", ped)

# Optimal price from linear demand: Q = a + b*P (where b < 0)
# Revenue = P * Q = a*P + b*P^2 -> P* = -a / (2*b)
a = lr_qty.intercept_
b = lr_qty.coef_[0]
if b < 0:
    p_opt = -a / (2 * b)
    print("Optimal price from weekly demand curve:", p_opt)
else:
    print("Slope is not negative")
