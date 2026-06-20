#!/usr/bin/env python3
"""Generate sample data for drift detection testing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

from pathlib import Path

np.random.seed(42)
random.seed(42)

SAMPLE_DIR = Path("sample_data")
SAMPLE_DIR.mkdir(exist_ok=True)

n_baseline = 5000
n_production = 3000

print("📊 Generating baseline data...")

baseline_age = np.random.normal(loc=35, scale=10, size=n_baseline).round(1)
baseline_income = np.random.normal(loc=60000, scale=15000, size=n_baseline).round(2)
baseline_balance = np.random.normal(loc=25000, scale=10000, size=n_baseline).round(2)
baseline_transactions = np.random.poisson(lam=15, size=n_baseline)
baseline_gender = np.random.choice(["M", "F"], size=n_baseline)
baseline_occupation = np.random.choice(["engineer", "teacher", "doctor", "manager", "student"],
                                   p=[0.3, 0.2, 0.15, 0.2, 0.15],
                                   size=n_baseline)
baseline_region = np.random.choice(["north", "south", "east", "west"],
                                p=[0.25, 0.25, 0.3, 0.2],
                                size=n_baseline)

baseline_labels = np.random.choice([0, 1], size=n_baseline, p=[0.7, 0.3])
baseline_preds = baseline_labels.copy()
mask = np.random.random(size=n_baseline) < 0.08
baseline_preds[mask] = 1 - baseline_preds[mask]

baseline_df = pd.DataFrame({
    "age": baseline_age,
    "income": baseline_income,
    "account_balance": baseline_balance,
    "transaction_count": baseline_transactions,
    "gender": baseline_gender,
    "occupation": baseline_occupation,
    "region": baseline_region,
})

baseline_df.to_csv(SAMPLE_DIR / "baseline_data.csv", index=False)
pd.DataFrame({"label": baseline_labels}).to_csv(SAMPLE_DIR / "baseline_labels.csv", index=False)
pd.DataFrame({"prediction": baseline_preds}).to_csv(SAMPLE_DIR / "baseline_predictions.csv", index=False)

print(f"✓ Baseline data generated: {n_baseline} samples")

print("\n📊 Generating production data...")

prod_age = np.random.normal(loc=42, scale=12, size=n_production).round(1)
prod_income = np.random.normal(loc=55000, scale=18000, size=n_production).round(2)
prod_balance = np.random.normal(loc=22000, scale=12000, size=n_production).round(2)
prod_transactions = np.random.poisson(lam=12, size=n_production)
prod_gender = np.random.choice(["M", "F"], size=n_production)
prod_occupation = np.random.choice(["engineer", "teacher", "doctor", "manager", "student"],
                                   p=[0.25, 0.25, 0.1, 0.25, 0.15],
                                   size=n_production)
prod_region = np.random.choice(["north", "south", "east", "west"],
                                p=[0.2, 0.3, 0.3, 0.2],
                                size=n_production)

start_date = datetime.now() - timedelta(days=30)
dates = [start_date + timedelta(days=random.randint(0, 30)) for _ in range(n_production)]
dates = sorted(dates)

prod_labels = np.random.choice([0, 1], size=n_production, p=[0.7, 0.3])
prod_preds = prod_labels.copy()
mask = np.random.random(size=n_production) < 0.15
prod_preds[mask] = 1 - prod_preds[mask]

production_df = pd.DataFrame({
    "age": prod_age,
    "income": prod_income,
    "account_balance": prod_balance,
    "transaction_count": prod_transactions,
    "gender": prod_gender,
    "occupation": prod_occupation,
    "region": prod_region,
    "timestamp": dates,
})

production_df.to_csv(SAMPLE_DIR / "production_data.csv", index=False)
pd.DataFrame({"label": prod_labels}).to_csv(SAMPLE_DIR / "production_labels.csv", index=False)
pd.DataFrame({"prediction": prod_preds}).to_csv(SAMPLE_DIR / "production_predictions.csv", index=False)

print(f"✓ Production data generated: {n_production} samples")

print("\n📊 Summary of generated data:")
print("\nBaseline features:")
print(f"  age: mean={baseline_age.mean():.1f}, std={baseline_age.std():.1f}")
print(f"  income: mean={baseline_income.mean():.0f}, std={baseline_income.std():.0f}")

print("\nProduction features:")
print(f"  age: mean={prod_age.mean():.1f}, std={prod_age.std():.1f}")
print(f"  income: mean={prod_income.mean():.0f}, std={prod_income.std():.0f}")

print(f"\n  ⚠️ Expected drift in 'age' (mean shift from 35 → 42)")
print(f"  ⚠️ Expected drift in 'income' (mean shift from 60k → 55k)")
print(f"  ✅ Expected stability in 'account_balance' (similar distribution)")

print("\n✅ Sample data generation complete!")
print(f"   Files saved to: {SAMPLE_DIR.absolute()}")
