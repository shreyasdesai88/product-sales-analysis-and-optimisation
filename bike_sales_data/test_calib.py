import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from generic_data_loader import GenericDataLoader

loader = GenericDataLoader("bike_sales_numeric_sp.csv")
df_all = loader.df

for model in loader.get_all_entities():
    df = df_all[df_all["Bike_Model"] == model].copy()
    date_col = "Date"
    price_col = "Price"
    target_kpi = "Total_Sales"
    
    df[date_col] = pd.to_datetime(df[date_col])
    df[price_col] = pd.to_numeric(df[price_col])
    df[target_kpi] = pd.to_numeric(df[target_kpi])
    
    t0 = df[date_col].min()
    t_days = (df[date_col] - t0).dt.total_seconds() / (24 * 3600)
    
    X = np.column_stack([df[price_col].values, t_days.values])
    y = df[target_kpi].values
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_p = poly.fit_transform(X)
    
    lr = LinearRegression()
    lr.fit(X_p, y)
    
    p_min = float(df[price_col].min())
    p_max = float(df[price_col].max())
    p_base = float(df[price_col].median())
    p_std = float(df[price_col].std())
    t_anchor = float(t_days.max())
    
    grid = np.linspace(p_min, p_max, 50)
    X_grid = np.column_stack([grid, np.full_like(grid, t_anchor)])
    X_grid_p = poly.transform(X_grid)
    raw_pred = lr.predict(X_grid_p)
    
    base_feat = poly.transform(np.array([[p_base, t_anchor]]))
    base_raw = float(lr.predict(base_feat)[0])
    
    # Let's test Demand Elasticity / Customer Acceptance curve:
    # Empirical price acceptance / willingness to pay:
    # Relative price deviation: rel_p = (grid - p_base) / p_std
    # Or rel_dev = (grid - p_base) / p_base
    rel_dev = (grid - p_base) / p_base
    
    # In economic pricing, the demand response curve has price elasticity around 1.2 to 1.8.
    # At baseline (rel_dev = 0), demand factor is 1.0.
    # When price is slightly raised, if baseline price was slightly conservative,
    # the optimal price occurs around 3% to 6% above median.
    # Let's test a demand retention function:
    # D(p) = exp(- elasticity * rel_dev - 0.5 * curvature * rel_dev^2)
    # If elasticity = 0.90 and curvature = 3.0:
    for (eps, curv) in [(0.85, 2.0), (0.90, 2.5), (0.92, 3.0)]:
        demand_factor = np.exp(-eps * rel_dev - 0.5 * curv * (rel_dev**2))
        opt_kpi = raw_pred * demand_factor
        idx = int(np.argmax(opt_kpi))
        best_p = grid[idx]
        best_val = opt_kpi[idx]
        uplift = (best_val - base_raw) / base_raw * 100
        print(f"[{model[:18]}] eps={eps}, curv={curv} -> Best Price: {best_p:.2f} (Base: {p_base:.2f}, Max: {p_max:.2f}), Uplift: {uplift:+.2f}%")
    print("-" * 60)
