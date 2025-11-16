"""
Example usage of Auto EDA library
"""

from auto_eda import AutoEDA
import pandas as pd
import numpy as np

# Example 1: Basic usage with a sample dataset
print("="*80)
print("EXAMPLE 1: Basic Usage")
print("="*80)

# Create a sample dataset with missing values
np.random.seed(42)
data = {
    'age': np.random.randint(18, 80, 100),
    'income': np.random.normal(50000, 15000, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'score': np.random.uniform(0, 100, 100)
}

# Introduce some missing values
df = pd.DataFrame(data)
df.loc[df.sample(10).index, 'age'] = np.nan
df.loc[df.sample(5).index, 'income'] = np.nan
df.loc[df.sample(3).index, 'category'] = np.nan

print(f"\nOriginal dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# Initialize AutoEDA
eda = AutoEDA(df=df)

# Run complete EDA workflow
# Note: This will ask for user input for output directory and train-test split
results = eda.run_complete_eda(
    auto_handle_missing=True,
    save_plots=True,
    save_report=True,
    get_split=False  # Set to True if you want train-test split
)

print(f"\nCleaned dataset shape: {results['cleaned_df'].shape}")
print(f"Strategies applied: {results['strategies_applied']}")

# Example 2: Load from file
print("\n" + "="*80)
print("EXAMPLE 2: Load from File")
print("="*80)

# Uncomment to use with your own CSV file:
# eda = AutoEDA(file_path='your_dataset.csv')
# results = eda.run_complete_eda()

# Example 3: Step-by-step usage
print("\n" + "="*80)
print("EXAMPLE 3: Step-by-Step Usage")
print("="*80)

eda2 = AutoEDA(df=df.copy())

# Step 1: Handle missing values
eda2.handle_missing_values(auto=True)

# Step 2: Perform EDA
eda2.perform_eda(save_plots=False)

# Step 3: Generate report (without saving)
eda2.generate_report(save_to_file=False)

# Step 4: Get cleaned dataframe
cleaned_df = eda2.get_cleaned_dataframe()
print(f"\nCleaned dataframe:\n{cleaned_df.head()}")

# Example 4: With train-test split
print("\n" + "="*80)
print("EXAMPLE 4: With Train-Test Split")
print("="*80)

# Add a target column for demonstration
df_with_target = df.copy()
df_with_target['target'] = (df_with_target['score'] > 50).astype(int)

eda3 = AutoEDA(df=df_with_target)
eda3.handle_missing_values(auto=True)
eda3.perform_eda(save_plots=False)

# Get train-test split (will ask for user input)
# X_train, X_test, y_train, y_test = eda3.get_train_test_split(target_column='target')

print("\nAll examples completed!")

