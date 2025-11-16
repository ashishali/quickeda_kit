"""
Test script to create a synthetic dataset and test the Auto EDA package
"""

import pandas as pd
import numpy as np
from auto_eda import AutoEDA

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("CREATING SYNTHETIC DATASET")
print("="*80)

# Create synthetic dataset with various features
n_samples = 1000

# Numeric features
age = np.random.randint(18, 80, n_samples)
income = np.random.normal(50000, 15000, n_samples)
income = np.clip(income, 20000, 120000)  # Clip outliers
score = np.random.uniform(0, 100, n_samples)
years_experience = np.random.exponential(5, n_samples)
years_experience = np.clip(years_experience, 0, 40)

# Categorical features
category = np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
city = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 
                        n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])
education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                            n_samples, p=[0.3, 0.4, 0.25, 0.05])

# Create target variable (for classification)
target = (score > 50).astype(int)

# Create dataframe
df = pd.DataFrame({
    'age': age,
    'income': income,
    'score': score,
    'years_experience': years_experience,
    'category': category,
    'city': city,
    'education': education,
    'target': target
})

# Introduce missing values strategically
# Age: 5% missing (random)
missing_age_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
df.loc[missing_age_indices, 'age'] = np.nan

# Income: 3% missing (random)
missing_income_indices = np.random.choice(df.index, size=int(0.03 * n_samples), replace=False)
df.loc[missing_income_indices, 'income'] = np.nan

# Category: 2% missing (random)
missing_category_indices = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
df.loc[missing_category_indices, 'category'] = np.nan

# Education: 4% missing (random)
missing_education_indices = np.random.choice(df.index, size=int(0.04 * n_samples), replace=False)
df.loc[missing_education_indices, 'education'] = np.nan

# Years_experience: 6% missing (random) - this will test median strategy
missing_exp_indices = np.random.choice(df.index, size=int(0.06 * n_samples), replace=False)
df.loc[missing_exp_indices, 'years_experience'] = np.nan

print(f"\nDataset created with {n_samples} samples and {len(df.columns)} columns")
print(f"\nMissing values before handling:")
print(df.isnull().sum())
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print("\n" + "="*80)
print("TESTING AUTO EDA PACKAGE")
print("="*80)

# Initialize AutoEDA
eda = AutoEDA(df=df)

# Run complete EDA workflow
# Note: This will be non-interactive for testing, but in real use it asks questions
print("\nRunning complete EDA workflow...")
print("(In interactive mode, you would be asked about output directory, etc.)\n")

# For testing, we'll run it step by step to avoid interactive prompts
# But first, let's set up output directory programmatically
eda.output_dir = "test_eda_output"
import os
os.makedirs(eda.output_dir, exist_ok=True)

# Step 1: Handle missing values
print("Step 1: Handling missing values...")
eda.handle_missing_values(auto=True)

# Step 2: Perform EDA
print("\nStep 2: Performing EDA analysis...")
eda.perform_eda(save_plots=True)

# Step 3: Generate report
print("\nStep 3: Generating report...")
eda.generate_report(save_to_file=True)

# Get results
cleaned_df = eda.get_cleaned_dataframe()
results = eda.get_results()

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"\nOriginal dataset shape: {df.shape}")
print(f"Cleaned dataset shape: {cleaned_df.shape}")
print(f"\nMissing values after handling:")
print(cleaned_df.isnull().sum().sum(), "missing values remaining")

print(f"\nStrategies applied for missing values:")
for col, strategy in eda.strategies_applied.items():
    print(f"  - {col}: {strategy}")

print(f"\nBasic Info:")
basic_info = results.get('basic_info', {})
print(f"  - Numeric columns: {len(basic_info.get('numeric_columns', []))}")
print(f"  - Categorical columns: {len(basic_info.get('categorical_columns', []))}")

print(f"\nOutlier Analysis:")
outliers = results.get('outliers', {})
for col, info in list(outliers.items())[:3]:
    print(f"  - {col}: {info['count']} outliers ({info['percentage']:.2f}%)")

print(f"\nCorrelation Analysis:")
correlations = results.get('correlations', {})
high_corr = correlations.get('high_correlation_pairs', [])
print(f"  - Found {len(high_corr)} highly correlated pairs (|r| > 0.7)")

print(f"\n✓ EDA Complete! Check '{eda.output_dir}' directory for visualizations and report.")
print("="*80)

