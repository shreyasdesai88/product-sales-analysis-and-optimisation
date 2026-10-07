"""
Organize Project Structure Utility
Categorizes and relocates files into clean, functional directories:
- data/       : Datasets and CSV files
- src/        : Core business logic, analytics, visualization and ML modules
- tests/      : Unit and integration tests
- scripts/    : Demo and diagnostic utility scripts
- docs/       : Architecture documentation, guides and notes
- deploy/     : Docker and container deployment configurations
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

ORGANIZATION_PLAN = {
    "data": [
        "bike_sales_numeric_sp.csv",
        "ipl_dataset_10000.csv",
    ],
    "src": [
        "generic_data_loader.py",
        "flexible_analytics_engine.py",
        "dynamic_visualizer.py",
        "ml_sales_optimizer.py",
    ],
    "tests": [
        "test_ml_sales_optimizer.py",
        "test_flexible_system.py",
        "test_dashboard_enhancements.py",
        "test_dashboard_fixes.py",
    ],
    "scripts": [
        "demo_ipl_comparison.py",
        "diagnose_issue.py",
    ],
    "docs": [
        "DASHBOARD_USER_GUIDE.md",
        "DATABASE_LOADING_FIX.md",
        "FLEXIBLE_SYSTEM_README.md",
        "FOLDER_CLEANUP.md",
        "IMPLEMENTATION_SUMMARY.md",
        "QUICK_START.md",
        "TEST_RESULTS.md",
        "TODO.md",
        "SETUP_COMPLETE.txt",
    ],
    "deploy": [
        "Dockerfile",
        "docker-compose.yml",
    ],
}


def organize():
    print("=" * 70)
    print("Starting Project Organization...")
    print("=" * 70)

    # 1. Create target directories
    for folder in ORGANIZATION_PLAN.keys():
        target_dir = ROOT / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created/verified folder: {folder}/")

    # 2. Create __init__.py in src and tests
    for pkg in ["src", "tests"]:
        init_file = ROOT / pkg / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Package marker."""\n', encoding="utf-8")
            print(f"Created {pkg}/__init__.py")

    # 3. Move/Copy files according to functional categories
    for folder, files in ORGANIZATION_PLAN.items():
        dest_dir = ROOT / folder
        for filename in files:
            src_file = ROOT / filename
            dest_file = dest_dir / filename
            if src_file.exists():
                shutil.copy2(str(src_file), str(dest_file))
                print(f"  [OK] {filename} -> {folder}/{filename}")
            elif dest_file.exists():
                print(f"  [EXISTS] {folder}/{filename} already in place")
            else:
                print(f"  [MISSING] {filename} not found")

    print("\nFile copying completed successfully.")


if __name__ == "__main__":
    organize()
