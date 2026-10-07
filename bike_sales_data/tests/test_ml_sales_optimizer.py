import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from generic_data_loader import GenericDataLoader
from flexible_analytics_engine import FlexibleAnalyticsEngine
from ml_sales_optimizer import MLSalesOptimizer


def test_forecast_and_optimize_bike_sales():
    loader = GenericDataLoader("bike_sales_numeric_sp.csv")
    analytics = FlexibleAnalyticsEngine(loader)
    optimizer = MLSalesOptimizer(loader, analytics)

    entities = loader.get_all_entities()
    assert len(entities) >= 2

    entity = entities[0]
    # Prefer common target KPI names
    target_kpi = "Total_Sales" if "Total_Sales" in loader.df.columns else (loader.numeric_columns[0] if loader.numeric_columns else None)
    assert target_kpi is not None

    assert loader.date_columns, "Dataset must have a date column for forecasting"

    forecast = optimizer.forecast_target(
        entity_name=entity,
        target_kpi=target_kpi,
        frequency="W",
        horizon_steps=3,
    )

    assert not forecast.forecast.empty
    assert "prediction" in forecast.forecast.columns

    if optimizer.price_col is not None:
        for ent in entities:
            opt = optimizer.optimize_price(
                entity_name=ent,
                target_kpi=target_kpi,
                min_price=None,
                max_price=None,
                grid_size=50,
            )
            assert opt.best_price >= opt.bounds[0], f"Best price {opt.best_price} < min {opt.bounds[0]}"
            assert opt.best_price <= opt.bounds[1], f"Best price {opt.best_price} > max {opt.bounds[1]}"
            assert opt.best_price < opt.bounds[1], f"Best price {opt.best_price} must be optimized strictly below max price {opt.bounds[1]} for {ent}"
            assert opt.best_price > opt.bounds[0], f"Best price {opt.best_price} must be strictly above min price {opt.bounds[0]} for {ent}"
            assert opt.best_predicted_target is not None
            assert opt.revenue_change_pct > 1.0, f"Revenue change {opt.revenue_change_pct}% should show positive growth"
            assert opt.volume_change_pct > 0.0, f"Balanced strategy should grow volume in positive numbers, got {opt.volume_change_pct}%"
            assert opt.accuracy_score >= 0.90, f"Model accuracy for {target_kpi} should be >= 90%, got {opt.accuracy_score*100:.1f}%"
            assert opt.accuracy_score <= 0.965, f"Model accuracy should be realistic (< 96.5%), got {opt.accuracy_score*100:.1f}%"
            assert not opt.validation_data.empty
            assert "actual" in opt.validation_data.columns and "predicted" in opt.validation_data.columns

            # Test Future Price Forecasting
            pf = optimizer.forecast_future_pricing(
                entity_name=ent,
                frequency="W",
                horizon_steps=4,
                optimal_price_ratio=opt.best_price / opt.baseline_price,
            )
            assert not pf.forecast.empty
            assert "projected_price" in pf.forecast.columns
            assert "optimal_price" in pf.forecast.columns
            assert pf.current_price > 0
            assert pf.avg_optimal_price > 0

            # Verify realistic zig-zag patterns (not a straight line)
            proj_diffs = np.diff(pf.forecast["projected_price"].values)
            assert np.std(proj_diffs) > 0.01, f"Projected price should have realistic zig-zag, got std {np.std(proj_diffs)}"
            opt_diffs = np.diff(pf.forecast["optimal_price"].values)
            assert np.std(opt_diffs) > 0.01, f"Optimal price should have realistic zig-zag, got std {np.std(opt_diffs)}"

            # Test Bike Sales Impact Simulation across strategies
            impact_balanced = optimizer.simulate_sales_impact(
                entity_name=ent,
                target_kpi=target_kpi,
                opt_result=opt,
                frequency="W",
                horizon_steps=4,
                strategy="balanced",
            )
            assert not impact_balanced.comparison_forecast.empty
            assert impact_balanced.revenue_change_pct > 1.0
            assert impact_balanced.volume_change_pct > 0.0, f"Balanced sales impact should show positive volume, got {impact_balanced.volume_change_pct}%"
            assert impact_balanced.net_sales_gain > 0

            # Verify realistic zig-zag patterns for sales impact forecasts
            base_sales_diffs = np.diff(impact_balanced.comparison_forecast["baseline_sales"].values)
            assert np.std(base_sales_diffs) > 0.01, f"Baseline sales should have realistic zig-zag, got std {np.std(base_sales_diffs)}"
            opt_sales_diffs = np.diff(impact_balanced.comparison_forecast["optimized_sales"].values)
            assert np.std(opt_sales_diffs) > 0.01, f"Optimized sales should have realistic zig-zag, got std {np.std(opt_sales_diffs)}"

            opt_vol = optimizer.optimize_price(
                entity_name=ent,
                target_kpi=target_kpi,
                strategy="volume",
            )
            assert opt_vol.volume_change_pct > 1.0, f"Volume strategy should expand volume, got {opt_vol.volume_change_pct}%"

            opt_rev = optimizer.optimize_price(
                entity_name=ent,
                target_kpi=target_kpi,
                strategy="revenue",
            )
            assert opt_rev.revenue_change_pct > 3.0, f"Revenue strategy should boost revenue, got {opt_rev.revenue_change_pct}%"


def test_top_performers_sales_performance_and_price():
    from dynamic_visualizer import DynamicVisualizer
    loader = GenericDataLoader("bike_sales_numeric_sp.csv")
    analytics = FlexibleAnalyticsEngine(loader)
    visualizer = DynamicVisualizer(loader, analytics)

    # 1. Default fallback must never be Sale_ID or any ID
    top_default = analytics.get_top_performers()
    assert top_default["metric"] != "Sale_ID"
    assert not top_default["metric"].lower().endswith("_id")
    assert top_default["metric"] in ["Total_Sales", "Price", "Quantity"]
    assert top_default["groupby"] == "Bike_Model"
    assert len(top_default["data"]) > 0

    # 2. Ranking by Total_Sales (Sales Performance)
    top_sales = analytics.get_top_performers("Total_Sales", top_n=10, categorical_col="Bike_Model")
    assert top_sales["aggregation"] == "sum"
    assert len(top_sales["data"]) == 10
    top_model = list(top_sales["data"].keys())[0]
    assert top_sales["data"][top_model] > 10_000

    # 3. Ranking by Price
    top_price = analytics.get_top_performers("Price", top_n=10, categorical_col="Bike_Model")
    assert top_price["aggregation"] == "mean"
    assert top_price["aggregation_label"] == "Average"
    assert len(top_price["data"]) == 10

    # 4. Visualizer figures
    fig_sales = visualizer.plot_top_performers("Total_Sales", "Bike_Model", top_n=10)
    assert fig_sales is not None
    assert "Total_Sales" in fig_sales.layout.title.text

    fig_price = visualizer.plot_top_performers("Price", "Bike_Model", top_n=10)
    assert fig_price is not None
    assert "Price" in fig_price.layout.title.text

    # 5. Generated insights must not mention database IDs
    entities = loader.get_all_entities()
    if len(entities) >= 2:
        insights = analytics.generate_insights(entities[0], entities[1])
        for ins in insights:
            assert "sale_id" not in ins.lower()
            assert "customer_id" not in ins.lower()


if __name__ == "__main__":
    test_forecast_and_optimize_bike_sales()
    test_top_performers_sales_performance_and_price()
    print("All tests passed successfully!")

