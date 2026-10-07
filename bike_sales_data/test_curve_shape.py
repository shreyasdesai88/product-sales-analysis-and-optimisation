import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from generic_data_loader import GenericDataLoader

loader = GenericDataLoader("bike_sales_numeric_sp.csv")
df = loader.df[loader.df["Bike_Model"] == "KTM 390 Adventure"].copy()

df["Date"] = pd.to_datetime(df["Date"])
df["Price"] = pd.to_numeric(df["Price"])
df["Total_Sales"] = pd.to_numeric(df["Total_Sales"])

t0 = df["Date"].min()
t_days = (df["Date"] - t0).dt.total_seconds() / (24 * 3600)
X = np.column_stack([df["Price"].values, t_days.values])
y = df["Total_Sales"].values

poly = PolynomialFeatures(degree=2, include_bias=False)
X_p = poly.fit_transform(X)

lr = LinearRegression()
lr.fit(X_p, y)

p_min = float(df["Price"].min())
p_max = float(df["Price"].max())
p_base = float(df["Price"].median())
t_anchor = float(t_days.max())

grid = np.linspace(p_min, p_max, 11)
X_grid = np.column_stack([grid, np.full_like(grid, t_anchor)])
X_grid_p = poly.transform(X_grid)
raw_pred = lr.predict(X_grid_p)

rel_dev = (grid - p_base) / p_base
demand_factor = np.exp(-0.90 * rel_dev - 0.5 * 2.5 * (rel_dev**2))
opt_kpi = raw_pred * demand_factor

for p, raw, dfact, opt in zip(grid, raw_pred, demand_factor, opt_kpi):
    print(f"Price: {p:9.2f} | Raw: {raw:9.2f} | Demand: {dfact:.4f} | Opt KPI: {opt:9.2f}")
