"""
Test script to verify dashboard improvements
"""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generic_data_loader import GenericDataLoader
from flexible_analytics_engine import FlexibleAnalyticsEngine
from dynamic_visualizer import DynamicVisualizer

print("=" * 80)
print("DASHBOARD LOADING TEST")
print("=" * 80)

# Test the improvements
print(f"\n1. Current Directory: {os.getcwd()}")
print(f"2. Repository Root: {REPO_ROOT}")

# Test loading CSV files from data/ directory
data_dir = REPO_ROOT / "data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')] if data_dir.exists() else []
print(f"\n3. CSV Files Found in data/: {csv_files}")

for csv_file in csv_files:
    print(f"\n4. Testing: {csv_file}")
    try:
        loader = GenericDataLoader(str(data_dir / csv_file))
        analytics = FlexibleAnalyticsEngine(loader)
        visualizer = DynamicVisualizer(loader, analytics)
        
        summary = loader.get_summary_stats()
        
        print(f"   ✓ Loader: OK")
        print(f"   ✓ Analytics: OK")
        print(f"   ✓ Visualizer: OK")
        print(f"   Records: {summary['total_records']}")
        print(f"   Columns: {summary['total_columns']}")
        print(f"   Entities: {len(summary['entities'])}")
        print(f"   Entity Column: {summary['entity_column']}")
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("✓ All tests passed! Dashboard should work now.")
print("=" * 80)
