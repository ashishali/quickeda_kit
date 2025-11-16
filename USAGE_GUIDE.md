# Auto EDA - Usage Guide

A comprehensive guide on how to use the Auto EDA package for automatic Exploratory Data Analysis.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Main Functions](#main-functions)
3. [Complete Workflows](#complete-workflows)
4. [Tips & Best Practices](#tips--best-practices)
5. [Common Questions](#common-questions)
6. [Quick Reference](#quick-reference)

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Basic Import

```python
from auto_eda import AutoEDA
import pandas as pd
```

### Initialize AutoEDA

You can initialize AutoEDA in two ways:

**Option 1: With a DataFrame**
```python
df = pd.read_csv('your_data.csv')
eda = AutoEDA(df=df)
```

**Option 2: Directly from file**
```python
eda = AutoEDA(file_path='your_data.csv')
# Supports: .csv, .xlsx, .xls, .json
```

---

## Main Functions

### 1. `run_complete_eda()` - The One-Command Solution

**What it does:** Runs the entire EDA workflow automatically - handles missing values, performs analysis, generates visualizations, creates reports, and optionally splits data.

**Usage:**
```python
results = eda.run_complete_eda(
    auto_handle_missing=True,  # Automatically fix missing values
    save_plots=True,            # Save charts and graphs
    save_report=True,           # Save text report
    get_split=False,            # Do train-test split? (True/False)
    target_column='target'       # Which column is your target? (optional)
)
```

**What you get back:**
```python
{
    'cleaned_df': DataFrame,      # Your cleaned data, ready to use
    'original_df': DataFrame,     # Original data (backup)
    'results': {...},              # All analysis results
    'strategies_applied': {...},   # How missing values were handled
    'split_data': (X_train, X_test, y_train, y_test)  # If get_split=True
}
```

**Example:**
```python
from auto_eda import AutoEDA
import pandas as pd

# Load your data
df = pd.read_csv('my_data.csv')

# Initialize
eda = AutoEDA(df=df)

# Run everything (one command!)
results = eda.run_complete_eda()

# Get your cleaned data
cleaned_data = results['cleaned_df']

# Now you can build models on cleaned_data!
```

---

### 2. `handle_missing_values()` - Fix Missing Data

**What it does:** Intelligently handles missing values in your dataset. Automatically decides whether to use mean, median, mode, drop, or fill strategies.

**Usage:**
```python
# Automatic mode (recommended)
cleaned_df, strategies = eda.handle_missing_values(auto=True)

# Interactive mode (asks you what to do for each column)
cleaned_df, strategies = eda.handle_missing_values(auto=False)
```

**What it does automatically:**
- **Numeric columns:** Uses mean (normal data) or median (skewed/outliers)
- **Categorical columns:** Uses mode (most frequent value)
- **High missing (>50%):** Drops the column
- **Shows you:** What strategy was applied to each column

**Example:**
```python
eda = AutoEDA(df=df)

# Handle missing values automatically
cleaned_df, strategies = eda.handle_missing_values(auto=True)

# See what strategies were used
print(strategies)
# Output: {'age': 'mean', 'income': 'median', 'category': 'mode'}

# Check missing values are gone
print(f"Missing values before: {df.isnull().sum().sum()}")
print(f"Missing values after: {cleaned_df.isnull().sum().sum()}")
```

---

### 3. `perform_eda()` - Run Comprehensive Analysis

**What it does:** Performs complete exploratory data analysis including statistics, outlier detection, correlations, distributions, and categorical analysis.

**Usage:**
```python
results = eda.perform_eda(save_plots=True)  # Set to False to skip saving plots
```

**What you get in results:**
- `basic_info`: Dataset shape, columns, memory usage
- `descriptive_stats`: Mean, median, std, min, max for all columns
- `outliers`: Outlier counts and percentages for each column
- `correlations`: Correlation matrix and highly correlated pairs
- `distributions`: Skewness, kurtosis, normality tests
- `categorical`: Value counts, most frequent values

**Example:**
```python
eda = AutoEDA(df=df)

# Perform EDA analysis
results = eda.perform_eda(save_plots=True)

# Check outliers
print("Outliers found:")
for col, info in results['outliers'].items():
    print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")

# Check highly correlated features
high_corr = results['correlations']['high_correlation_pairs']
print("\nHighly correlated pairs:")
for pair in high_corr:
    print(f"{pair['column1']} <-> {pair['column2']}: {pair['correlation']:.3f}")

# Check distributions
print("\nDistribution analysis:")
for col, dist_info in results['distributions'].items():
    print(f"{col}: skewness={dist_info['skewness']:.3f}, "
          f"kurtosis={dist_info['kurtosis']:.3f}")
```

---

### 4. `generate_report()` - Create Detailed Report

**What it does:** Generates a comprehensive text report with all findings, statistics, and recommendations.

**Usage:**
```python
report_text = eda.generate_report(save_to_file=True)
```

**Note:** You must run `perform_eda()` first before generating a report.

**Example:**
```python
eda = AutoEDA(df=df)

# First perform EDA
eda.perform_eda(save_plots=True)

# Then generate report
report = eda.generate_report(save_to_file=True)
# Saves to: eda_output/eda_report.txt (or your specified output directory)

# Report is also printed to console
print(report)
```

---

### 5. `get_train_test_split()` - Split Your Data

**What it does:** Splits your dataset into training and testing sets for machine learning, with interactive prompts.

**Usage:**
```python
# Fully interactive (asks you questions)
X_train, X_test, y_train, y_test = eda.get_train_test_split()

# Or specify everything yourself
X_train, X_test, y_train, y_test = eda.get_train_test_split(
    target_column='target',
    test_size=0.2,
    random_state=42
)
```

**Example:**
```python
eda = AutoEDA(df=df)

# Interactive mode - will ask you:
# 1. Which column is the target? (shows list)
# 2. What test size? (default: 0.2)
# 3. Random state? (default: 42)
X_train, X_test, y_train, y_test = eda.get_train_test_split()

# Or specify everything
X_train, X_test, y_train, y_test = eda.get_train_test_split(
    target_column='target',
    test_size=0.3,
    random_state=42
)

print(f"Train: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# Now ready for model training!
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

---

### 6. `get_cleaned_dataframe()` - Get Cleaned Data

**What it does:** Returns the cleaned DataFrame after missing value handling.

**Usage:**
```python
cleaned_df = eda.get_cleaned_dataframe()
```

**Example:**
```python
eda = AutoEDA(df=df)

# Handle missing values first
eda.handle_missing_values()

# Get cleaned data
cleaned_df = eda.get_cleaned_dataframe()

# Now cleaned_df has no missing values!
print(cleaned_df.isnull().sum().sum())  # Should be 0
```

---

### 7. `get_results()` - Get All Analysis Results

**What it does:** Returns all EDA analysis results as a dictionary.

**Usage:**
```python
results = eda.get_results()
```

**Example:**
```python
eda = AutoEDA(df=df)

# Perform EDA first
eda.perform_eda()

# Get all results
results = eda.get_results()

# Access specific results
print("Basic info:", results['basic_info'])
print("Outliers:", results['outliers'])
print("Correlations:", results['correlations'])
print("Distributions:", results['distributions'])
print("Categorical:", results['categorical'])
```

---

## Complete Workflows

### Workflow 1: Quick Start (Recommended for Beginners)

**Best for:** When you want everything done automatically with minimal code.

```python
from auto_eda import AutoEDA
import pandas as pd

# 1. Load your data
df = pd.read_csv('my_data.csv')

# 2. Initialize
eda = AutoEDA(df=df)

# 3. Run everything (one command!)
results = eda.run_complete_eda()

# 4. Get your cleaned data
cleaned_data = results['cleaned_df']

# 5. Build your models!
from sklearn.ensemble import RandomForestClassifier

# If you want train-test split, use:
results = eda.run_complete_eda(get_split=True, target_column='target')
X_train, X_test, y_train, y_test = results['split_data']

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

---

### Workflow 2: Step-by-Step (More Control)

**Best for:** When you want to control each step and see what's happening.

```python
from auto_eda import AutoEDA
import pandas as pd

# 1. Load data
df = pd.read_csv('my_data.csv')

# 2. Initialize
eda = AutoEDA(df=df)

# 3. Set output directory (optional)
eda.output_dir = "my_eda_output"

# 4. Handle missing values
eda.handle_missing_values(auto=True)

# 5. Perform EDA
results = eda.perform_eda(save_plots=True)

# 6. Generate report
eda.generate_report(save_to_file=True)

# 7. Get train-test split
X_train, X_test, y_train, y_test = eda.get_train_test_split(
    target_column='target',
    test_size=0.2
)

# 8. Get cleaned data
cleaned_df = eda.get_cleaned_dataframe()

# 9. Build models
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

---

### Workflow 3: With Train-Test Split

**Best for:** When you need train-test split as part of your workflow.

```python
from auto_eda import AutoEDA
import pandas as pd

# Load data
df = pd.read_csv('my_data.csv')

# Initialize
eda = AutoEDA(df=df)

# Run complete workflow with train-test split
results = eda.run_complete_eda(
    auto_handle_missing=True,
    save_plots=True,
    save_report=True,
    get_split=True,              # Enable train-test split
    target_column='target'        # Specify target column
)

# Extract split data
X_train, X_test, y_train, y_test = results['split_data']

# Get cleaned data
cleaned_df = results['cleaned_df']

# Build and train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
```

---

### Workflow 4: Custom Analysis

**Best for:** When you want to access specific analysis results.

```python
from auto_eda import AutoEDA
import pandas as pd

# Load data
df = pd.read_csv('my_data.csv')

# Initialize
eda = AutoEDA(df=df)

# Handle missing values
eda.handle_missing_values(auto=True)

# Perform EDA
results = eda.perform_eda(save_plots=True)

# Access specific results
basic_info = results['basic_info']
print(f"Dataset shape: {basic_info['shape']}")
print(f"Numeric columns: {basic_info['numeric_columns']}")

# Analyze outliers
outliers = results['outliers']
for col, info in outliers.items():
    if info['percentage'] > 5:  # More than 5% outliers
        print(f"Warning: {col} has {info['percentage']:.2f}% outliers")

# Check correlations
correlations = results['correlations']
high_corr = correlations['high_correlation_pairs']
if high_corr:
    print("Highly correlated features found:")
    for pair in high_corr:
        print(f"  {pair['column1']} <-> {pair['column2']}: {pair['correlation']:.3f}")

# Check distributions
distributions = results['distributions']
for col, dist_info in distributions.items():
    if abs(dist_info['skewness']) > 1:
        print(f"{col} is skewed (skewness: {dist_info['skewness']:.3f})")
```

---

## Tips & Best Practices

### 1. Start Simple
Begin with `run_complete_eda()` - it does everything automatically. You can always customize later.

### 2. Use Automatic Missing Value Handling
Set `auto=True` for `handle_missing_values()` - the package intelligently chooses the best strategy.

### 3. Set Output Directory
If you want to control where files are saved:
```python
eda = AutoEDA(df=df)
eda.output_dir = "my_custom_output"  # Set before running
eda.run_complete_eda()
```

### 4. Check Strategies Applied
Always review how missing values were handled:
```python
results = eda.run_complete_eda()
print(results['strategies_applied'])
```

### 5. Save Plots for Large Datasets
For large datasets, you might want to skip saving plots to speed things up:
```python
results = eda.run_complete_eda(save_plots=False)
```

### 6. Use Train-Test Split After EDA
Always split your data after EDA, not before, to avoid data leakage.

### 7. Review the Report
The generated report contains valuable insights and recommendations - always read it!

---

## Common Questions

### Q: Do I need to run functions in a specific order?

**A:** If you use `run_complete_eda()`, no - it handles everything. If doing step-by-step:
1. First: `handle_missing_values()`
2. Then: `perform_eda()`
3. Then: `generate_report()`
4. Finally: `get_train_test_split()` (if needed)

### Q: Can I use it without saving files?

**A:** Yes! Set `save_plots=False` and `save_report=False`:
```python
results = eda.run_complete_eda(save_plots=False, save_report=False)
```

### Q: How do I see what strategy was chosen for missing values?

**A:** Check `eda.strategies_applied` or `results['strategies_applied']`:
```python
results = eda.run_complete_eda()
print(results['strategies_applied'])
```

### Q: Can I use it on Excel files?

**A:** Yes! Use `file_path` parameter:
```python
eda = AutoEDA(file_path='data.xlsx')
```

### Q: What if I want to manually choose strategies for missing values?

**A:** Use interactive mode:
```python
eda.handle_missing_values(auto=False)
# Will ask you for each column
```

### Q: Can I access individual analysis results?

**A:** Yes! Use `get_results()` or access from the returned dictionary:
```python
results = eda.run_complete_eda()
outliers = results['results']['outliers']
correlations = results['results']['correlations']
```

### Q: How do I know if my data is ready for modeling?

**A:** After running `run_complete_eda()`, check:
- No missing values: `results['cleaned_df'].isnull().sum().sum() == 0`
- Review the report for recommendations
- Check for high correlations that might need feature selection

---

## Quick Reference

### Initialization
```python
from auto_eda import AutoEDA
import pandas as pd

# With DataFrame
eda = AutoEDA(df=df)

# From file
eda = AutoEDA(file_path='data.csv')
```

### Complete Workflow
```python
# Everything at once
results = eda.run_complete_eda()

# With train-test split
results = eda.run_complete_eda(get_split=True, target_column='target')
```

### Step-by-Step
```python
# Handle missing values
eda.handle_missing_values(auto=True)

# Perform EDA
eda.perform_eda(save_plots=True)

# Generate report
eda.generate_report(save_to_file=True)

# Train-test split
X_train, X_test, y_train, y_test = eda.get_train_test_split(target_column='target')
```

### Access Results
```python
# Get cleaned data
cleaned_df = eda.get_cleaned_dataframe()

# Get all results
results = eda.get_results()

# From run_complete_eda
results = eda.run_complete_eda()
cleaned_df = results['cleaned_df']
all_results = results['results']
strategies = results['strategies_applied']
```

### Common Parameters
```python
# run_complete_eda parameters
auto_handle_missing=True   # Auto-handle missing values
save_plots=True             # Save visualizations
save_report=True            # Save text report
get_split=False             # Do train-test split
target_column='target'      # Target column name

# handle_missing_values parameters
auto=True                   # Automatic strategy selection

# perform_eda parameters
save_plots=True             # Save visualization plots

# get_train_test_split parameters
target_column='target'      # Target column name
test_size=0.2              # Test set proportion (0.0-1.0)
random_state=42            # Random seed
```

---

## Example: Complete End-to-End Usage

```python
from auto_eda import AutoEDA
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
df = pd.read_csv('my_dataset.csv')
print(f"Original shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# 2. Initialize AutoEDA
eda = AutoEDA(df=df)

# 3. Run complete EDA workflow
results = eda.run_complete_eda(
    auto_handle_missing=True,
    save_plots=True,
    save_report=True,
    get_split=True,
    target_column='target'
)

# 4. Get split data
X_train, X_test, y_train, y_test = results['split_data']

# 5. Get cleaned data
cleaned_df = results['cleaned_df']
print(f"\nCleaned shape: {cleaned_df.shape}")
print(f"Missing values: {cleaned_df.isnull().sum().sum()}")

# 6. Review strategies applied
print("\nMissing value strategies:")
for col, strategy in results['strategies_applied'].items():
    print(f"  {col}: {strategy}")

# 7. Check analysis results
print("\nOutlier summary:")
for col, info in results['results']['outliers'].items():
    if info['count'] > 0:
        print(f"  {col}: {info['count']} outliers ({info['percentage']:.2f}%)")

# 8. Build and train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 9. Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy:.3f}")

print("\n✓ Complete workflow finished!")
print("✓ Check output directory for visualizations and report")
```

---

## Summary

The Auto EDA package provides a complete solution for exploratory data analysis:

1. **One Command Solution:** `run_complete_eda()` does everything
2. **Intelligent Missing Value Handling:** Automatically chooses best strategies
3. **Comprehensive Analysis:** Statistics, outliers, correlations, distributions
4. **Visualizations:** Automatic generation of charts and graphs
5. **Detailed Reports:** Complete summary with recommendations
6. **Model-Ready Data:** Cleaned dataset ready for machine learning

Start with `run_complete_eda()` for the simplest experience, or use step-by-step functions for more control!

