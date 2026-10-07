from generic_data_loader import GenericDataLoader
from ml_sales_optimizer import MLSalesOptimizer

loader = GenericDataLoader("bike_sales_numeric_sp.csv")
opt = MLSalesOptimizer(loader)
for model in loader.get_all_entities():
    res = opt.optimize_price(model, "Total_Sales")
    print(f"\nModel: {model}")
    print(f"Bounds: min={res.bounds[0]:.2f}, max={res.bounds[1]:.2f}")
    print(f"Best Price: {res.best_price:.2f}")
    print(f"Baseline Price: {res.baseline_price:.2f}")
    print(f"Is Best Price == Max Price? {abs(res.best_price - res.bounds[1]) < 1e-4}")
