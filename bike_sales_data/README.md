# 🚲 Bike Sales Analytics & AI Optimization Dashboard

A modular, dataset-agnostic business intelligence and machine learning analytics platform. Automatically adapts to any CSV dataset structure, computes comparative KPIs, predicts sales trajectories, and optimizes pricing strategy.

---

## 📁 Project Architecture & Folder Structure

The project has been organized into clear, functional directories:

```
bike_sales_data/
├── data/                             # Datasets and raw data files
│   ├── bike_sales_numeric_sp.csv     # 10,000+ bike transaction records
│   └── ipl_dataset_10000.csv         # 10,000+ IPL match statistics
│
├── src/                              # Core application business logic
│   ├── __init__.py                   # Package marker
│   ├── generic_data_loader.py        # Universal CSV ingestion & schema auto-detector
│   ├── flexible_analytics_engine.py  # Analytics, entity comparison & KPI engine
│   ├── dynamic_visualizer.py         # Plotly visualization & theme engine
│   └── ml_sales_optimizer.py         # ML demand forecasting & pricing optimizer
│
├── tests/                            # Comprehensive automated test suites
│   ├── __init__.py                   # Test package marker
│   ├── test_ml_sales_optimizer.py    # Unit tests for ML forecasting & pricing
│   ├── test_flexible_system.py       # Cross-dataset adaptability tests
│   ├── test_dashboard_enhancements.py# KPI selection & filtering tests
│   └── test_dashboard_fixes.py       # Path and loader regression tests
│
├── scripts/                          # Utility, diagnostic and demo scripts
│   ├── demo_ipl_comparison.py        # Standalone comparison demo (IPL dataset)
│   └── diagnose_issue.py             # Diagnostic tool for database/path issues
│
├── docs/                             # User guides, technical specs & documentation
│   ├── DASHBOARD_USER_GUIDE.md       # Interactive dashboard user manual
│   ├── DATABASE_LOADING_FIX.md       # Data loader architecture & fixes
│   ├── FLEXIBLE_SYSTEM_README.md     # Analytics engine API documentation
│   ├── FOLDER_CLEANUP.md             # File cleanup history
│   ├── IMPLEMENTATION_SUMMARY.md     # Technical implementation details
│   ├── QUICK_START.md                # Quickstart cheatsheet
│   ├── TEST_RESULTS.md               # Test coverage & verification results
│   ├── TODO.md                       # Roadmap & pending items
│   └── SETUP_COMPLETE.txt            # Environment confirmation
│
├── deploy/                           # Containerization & deployment configs
│   ├── Dockerfile                    # Container build specification
│   └── docker-compose.yml            # Multi-service container orchestration
│
├── dynamic_dashboard.py              # Main Streamlit application entrypoint
├── requirements.txt                  # Production dependencies
├── requirements-working.txt          # Frozen dependency lockfile
├── README.md                         # Project overview and architecture guide
└── .gitignore                        # Git ignore patterns
```

---

## 🛠️ Functional Component Breakdown

| Directory / File | Function & Responsibility |
| :--- | :--- |
| **`data/`** | Centralized storage for CSV datasets (`bike_sales_numeric_sp.csv`, `ipl_dataset_10000.csv`). Automatically scanned by the data loader. |
| **`src/generic_data_loader.py`** | Detects column types (numeric, categorical, temporal), primary entity axis, and resolves paths across both `data/` and parent folders. |
| **`src/flexible_analytics_engine.py`** | Computes descriptive statistics, entity comparisons, correlations, top performers, and automated data-driven insights. |
| **`src/dynamic_visualizer.py`** | Builds high-contrast, accessible Plotly visualizations (time series, distributions, heatmaps, horizontal ranking bars). |
| **`src/ml_sales_optimizer.py`** | Machine learning engine that models price elasticity, volume-price interactions, realistic trajectory wave-forms, and recommends revenue/volume-maximizing price targets with +90% predictive accuracy. |
| **`dynamic_dashboard.py`** | Streamlit web application providing interactive tabs: Overview, Executive Comparison, Visual Explorer, Metric Breakdown, Trends, Insights, and AI Optimization. |
| **`tests/`** | Automated unit tests covering all analytics calculations, ML forecasting, top performers ranking, and error handling. |
| **`scripts/`** | Standalone verification scripts to demonstrate multi-dataset compatibility or diagnose path configurations. |
| **`docs/`** | Comprehensive documentation, architecture notes, user guides, and troubleshooting steps. |
| **`deploy/`** | Docker recipes for containerized deployment. |

---

## 🚀 Quick Start Guide

### 1. Installation
Ensure Python 3.9+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launching the Dashboard
Start the interactive Streamlit application:
```bash
streamlit run dynamic_dashboard.py
```

### 3. Running the Test Suite
Run any of the tests in the `tests/` directory:
```bash
python tests/test_ml_sales_optimizer.py
python tests/test_flexible_system.py
```

### 4. Running Demos
Test the engine on alternate datasets (e.g. IPL match data):
```bash
python scripts/demo_ipl_comparison.py
```
