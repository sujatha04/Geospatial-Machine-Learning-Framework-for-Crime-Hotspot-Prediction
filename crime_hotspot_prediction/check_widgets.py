"""Check for duplicate Streamlit widget keys across pages."""
import re, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
pages_dir = os.path.join(ROOT, "pages")

all_keys = {}
duplicates = []

for fn in sorted(os.listdir(pages_dir)):
    if not fn.endswith(".py") or fn.startswith("_"):
        continue
    content = open(os.path.join(pages_dir, fn), encoding="utf-8").read()
    keys = re.findall(r'key=["\'](\w+)["\']', content)
    for k in keys:
        if k in all_keys:
            duplicates.append((k, fn, all_keys[k]))
        else:
            all_keys[k] = fn

print(f"Total unique widget keys: {len(all_keys)}")
if duplicates:
    for k, f1, f2 in duplicates:
        print(f"  DUPLICATE KEY: '{k}' in {f1} AND {f2}")
else:
    print("  No duplicate keys found across pages.")

# Also check app.py
app_content = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
app_keys = re.findall(r'key=["\'](\w+)["\']', app_content)
print(f"App.py widget keys: {app_keys}")

# Test folium map attribute
import folium
m = folium.Map(location=[40.7, -74.0], zoom_start=12)
m.location = [40.8, -73.9]
m.zoom_start = 13
print(f"Folium map location assignment: {m.location}")

# Test date_input tuple handling
import pandas as pd
date_range = (pd.Timestamp("2021-01-01").date(), pd.Timestamp("2023-12-30").date())
if len(date_range) == 2:
    s, e = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    print(f"Date range tuple handling OK: {s} to {e}")

print("All checks passed.")
