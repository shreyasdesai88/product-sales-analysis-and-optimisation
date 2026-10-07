import numpy as np
import pandas as pd

df = pd.read_csv("bike_sales_numeric_sp.csv")
for model in df["Bike_Model"].unique():
    mdf = df[df["Bike_Model"] == model]
    prices = mdf["Price"].values
    p_med = float(np.median(prices))
    p_mean = float(np.mean(prices))
    p_std = float(np.std(prices))
    p_min = float(np.min(prices))
    p_max = float(np.max(prices))
    
    print(f"\nModel: {model}")
    print(f"Price: min={p_min:.1f}, median={p_med:.1f}, mean={p_mean:.1f}, max={p_max:.1f}, std={p_std:.1f}")
    
    # Test different elasticity / demand response formulas:
    grid = np.linspace(p_min, p_max, 100)
    
    # Formula 1: Constant elasticity around baseline
    # Revenue(p) = p * (p / p_med)**(-eps) -> has no interior peak if eps is constant
    
    # Formula 2: Linear-demand / Quadratic revenue (Degree 2 polynomial)
    # Q(p) = Q0 * [1 - eps * (p - p_med) / p_med]
    # Revenue(p) = p * Q(p) = p * Q0 * [1 + eps - eps * p / p_med]
    # Peak is at p* = (1 + eps) / (2 * eps) * p_med
    for eps in [1.1, 1.2, 1.3, 1.4]:
        p_opt_quad = (1.0 + eps) / (2.0 * eps) * p_med
        rev_base = p_med * 1.0
        rev_opt = p_opt_quad * (1.0 - eps * (p_opt_quad - p_med) / p_med)
        uplift = (rev_opt - rev_base) / rev_base * 100
        print(f"  Quadratic eps={eps:.1f}: p*={p_opt_quad:.1f} (vs med: {p_opt_quad - p_med:+.1f}, uplift={uplift:+.2f}%)")
