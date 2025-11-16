# Complete API Reference with Examples

This document explains every function in the Auto EDA package with detailed examples.

---

## Table of Contents

1. [AutoEDA Class (Main Class)](#autoeda-class-main-class)
2. [MissingValueHandler Class](#missingvaluehandler-class)
3. [EDAAnalyzer Class](#edaanalyzer-class)
4. [ReportGenerator Class](#reportgenerator-class)

---

## AutoEDA Class (Main Class)

The main class that orchestrates the entire EDA workflow.

### `__init__(df=None, file_path=None)`

**Purpose:** Initialize the AutoEDA object with a dataset.

**Parameters:**
- `df` (pandas.DataFrame, optional): Input dataframe
- `file_path` (str, optional): Path to CSV/Excel/JSON file to load

**Example 1: Initialize with DataFrame**
```python
import pandas as pd
from auto_eda import AutoEDA

# Create a sample dataframe
df = pd.DataFrame({
    'age': [25, 30, 35, None, 40],
    'income': [50000, 60000, None, 70000, 80000],
    'city': ['NYC', 'LA', None, 'Chicago', 'NYC']
})

# Initialize AutoEDA
eda = AutoEDA(df=df)
print(f"Dataset shape: {eda.df.shape}")
```

**Example 2: Initialize with File Path**
```python
from auto_eda import AutoEDA

# Load from CSV
eda = AutoEDA(file_path='data.csv')

# Load from Excel
eda = AutoEDA(file_path='data.xlsx')

# Load from JSON
eda = AutoEDA(file_path='data.json')
```

**Example 3: Error Handling**
```python
# This will raise ValueError
try:
    eda = AutoEDA()  # No df or file_path provided
except ValueError as e:
    print(f"Error: {e}")
```

---

### `handle_missing_values(auto=True)`

**Purpose:** Handle missing values in the dataset with intelligent strategies.

**Parameters:**
- `auto` (bool): If True, automatically decide strategies. If False, ask user interactively.

**Returns:**
- `cleaned_df` (DataFrame): DataFrame with missing values handled
- `strategies_applied` (dict): Dictionary mapping columns to strategies used

**Example 1: Automatic Mode (Recommended)**
```python
import pandas as pd
import numpy as np
from auto_eda import AutoEDA

# Create dataset with missing values
df = pd.DataFrame({
    'age': [25, 30, None, 35, None, 40],
    'income': [50000, None, 60000, None, 70000, 80000],
    'category': ['A', 'B', None, 'A', 'B', None]
})

eda = AutoEDA(df=df)

# Automatically handle missing values
cleaned_df, strategies = eda.handle_missing_values(auto=True)

print("Strategies applied:")
for col, strategy in strategies.items():
    print(f"  {col}: {strategy}")

print(f"\nMissing values before: {df.isnull().sum().sum()}")
print(f"Missing values after: {cleaned_df.isnull().sum().sum()}")
```

**Example 2: Interactive Mode**
```python
eda = AutoEDA(df=df)

# This will ask you for each column's strategy
cleaned_df, strategies = eda.handle_missing_values(auto=False)
# You'll be prompted: "Choose strategy for 'age' (mean/median/mode/drop/forward_fill/backward_fill):"
```

**Example 3: Check What Strategies Were Applied**
```python
eda = AutoEDA(df=df)
cleaned_df, strategies = eda.handle_missing_values(auto=True)

# Access strategies later
print(eda.strategies_applied)
# Output: {'age': 'mean', 'income': 'mean', 'category': 'mode'}
```

---

### `perform_eda(save_plots=True)`

**Purpose:** Perform comprehensive exploratory data analysis.

**Parameters:**
- `save_plots` (bool): Whether to save visualization plots to output directory

**Returns:**
- `results` (dict): Dictionary containing all EDA results

**Example 1: Basic EDA**
```python
import pandas as pd
from auto_eda import AutoEDA

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000],
    'score': [85, 90, 75, 95, 88]
})

eda = AutoEDA(df=df)
eda.output_dir = "my_output"  # Set output directory first

# Perform EDA
results = eda.perform_eda(save_plots=True)

# Access results
print("Basic info:", results['basic_info'])
print("Descriptive stats:", results['descriptive_stats'])
print("Outliers:", results['outliers'])
print("Correlations:", results['correlations'])
```

**Example 2: EDA Without Saving Plots**
```python
eda = AutoEDA(df=df)

# Perform EDA without saving plots (faster)
results = eda.perform_eda(save_plots=False)

# You can still access all analysis results
print(results.keys())
# Output: dict_keys(['basic_info', 'descriptive_stats', 'distributions', 
#                    'outliers', 'correlations', 'categorical'])
```

**Example 3: Access Specific Analysis Results**
```python
eda = AutoEDA(df=df)
results = eda.perform_eda(save_plots=True)

# Get basic information
basic_info = results['basic_info']
print(f"Dataset shape: {basic_info['shape']}")
print(f"Numeric columns: {basic_info['numeric_columns']}")
print(f"Categorical columns: {basic_info['categorical_columns']}")

# Get outlier information
outliers = results['outliers']
for col, info in outliers.items():
    print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")

# Get correlation pairs
correlations = results['correlations']
high_corr = correlations['high_correlation_pairs']
for pair in high_corr:
    print(f"{pair['column1']} <-> {pair['column2']}: {pair['correlation']:.3f}")
```

---

### `generate_report(save_to_file=True)`

**Purpose:** Generate a comprehensive text report of all EDA findings.

**Parameters:**
- `save_to_file` (bool): Whether to save report to file

**Returns:**
- `report_text` (str): The generated report as a string

**Example 1: Generate and Save Report**
```python
import pandas as pd
from auto_eda import AutoEDA

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000],
    'category': ['A', 'B', 'A', 'B', 'A']
})

eda = AutoEDA(df=df)
eda.output_dir = "reports"  # Set output directory

# First perform EDA
eda.perform_eda(save_plots=True)

# Then generate report
report = eda.generate_report(save_to_file=True)
# Report is saved to: reports/eda_report.txt
```

**Example 2: Generate Report Without Saving**
```python
eda = AutoEDA(df=df)
eda.perform_eda(save_plots=False)

# Generate report but don't save (just print)
report = eda.generate_report(save_to_file=False)
print(report)  # Print to console
```

**Example 3: Access Report Content**
```python
eda = AutoEDA(df=df)
eda.perform_eda()
report = eda.generate_report(save_to_file=True)

# Report is a string, you can process it
lines = report.split('\n')
print(f"Report has {len(lines)} lines")
print("First 10 lines:")
print('\n'.join(lines[:10]))
```

---

### `get_train_test_split(target_column=None, test_size=None, random_state=None)`

**Purpose:** Split dataset into training and testing sets with interactive prompts.

**Parameters:**
- `target_column` (str, optional): Name of target column. If None, asks user.
- `test_size` (float, optional): Proportion of test set (0.0-1.0). If None, asks user.
- `random_state` (int, optional): Random seed. If None, asks user.

**Returns:**
- `X_train, X_test, y_train, y_test`: Split datasets

**Example 1: Fully Interactive (Recommended)**
```python
import pandas as pd
from auto_eda import AutoEDA

df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

eda = AutoEDA(df=df)

# This will ask you:
# 1. Which column is the target? (shows list)
# 2. What test size? (default: 0.2)
# 3. Random state? (default: 42)
X_train, X_test, y_train, y_test = eda.get_train_test_split()

print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")
```

**Example 2: Specify All Parameters (Non-Interactive)**
```python
eda = AutoEDA(df=df)

# Provide all parameters to skip prompts
X_train, X_test, y_train, y_test = eda.get_train_test_split(
    target_column='target',
    test_size=0.3,
    random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

**Example 3: Partial Parameters**
```python
eda = AutoEDA(df=df)

# Specify target, but let it ask for test_size and random_state
X_train, X_test, y_train, y_test = eda.get_train_test_split(
    target_column='target'
)
# Will still ask for test_size and random_state
```

---

### `run_complete_eda(auto_handle_missing=True, save_plots=True, save_report=True, get_split=False, target_column=None)`

**Purpose:** Run the complete EDA workflow in one command.

**Parameters:**
- `auto_handle_missing` (bool): Automatically handle missing values
- `save_plots` (bool): Save visualization plots
- `save_report` (bool): Save report to file
- `get_split` (bool): Whether to perform train-test split
- `target_column` (str, optional): Target column for split

**Returns:**
- `results` (dict): Dictionary with cleaned_df, original_df, results, strategies_applied, split_data

**Example 1: Complete Workflow (Most Common)**
```python
import pandas as pd
from auto_eda import AutoEDA

df = pd.read_csv('my_data.csv')

# One command does everything!
results = eda.run_complete_eda()

# Access results
cleaned_df = results['cleaned_df']
original_df = results['original_df']
eda_results = results['results']
strategies = results['strategies_applied']
```

**Example 2: Complete Workflow with Train-Test Split**
```python
eda = AutoEDA(df=df)

# Run everything including train-test split
results = eda.run_complete_eda(
    auto_handle_missing=True,
    save_plots=True,
    save_report=True,
    get_split=True,
    target_column='target'  # Optional: specify target
)

# Access split data
if results['split_data']:
    X_train, X_test, y_train, y_test = results['split_data']
```

**Example 3: Customized Workflow**
```python
eda = AutoEDA(df=df)

# Customize the workflow
results = eda.run_complete_eda(
    auto_handle_missing=False,  # Ask user for each missing value strategy
    save_plots=False,           # Don't save plots (faster)
    save_report=True,           # Still save report
    get_split=False             # Don't do train-test split
)
```

---

### `get_cleaned_dataframe()`

**Purpose:** Get the cleaned dataframe after EDA processing.

**Returns:**
- `DataFrame`: The cleaned dataframe

**Example:**
```python
import pandas as pd
from auto_eda import AutoEDA

df = pd.DataFrame({
    'age': [25, None, 35, 40],
    'income': [50000, 60000, None, 80000]
})

eda = AutoEDA(df=df)
eda.handle_missing_values()

# Get cleaned dataframe
cleaned_df = eda.get_cleaned_dataframe()
print(cleaned_df)
# All missing values are now filled
```

---

### `get_results()`

**Purpose:** Get all EDA analysis results.

**Returns:**
- `dict`: Dictionary containing all EDA results

**Example:**
```python
import pandas as pd
from auto_eda import AutoEDA

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000]
})

eda = AutoEDA(df=df)
eda.perform_eda()

# Get all results
results = eda.get_results()

# Access specific results
print("Basic info:", results['basic_info'])
print("Outliers:", results['outliers'])
print("Correlations:", results['correlations'])
print("Distributions:", results['distributions'])
```

---

## MissingValueHandler Class

Handles missing values with intelligent decision making.

### `__init__(df)`

**Purpose:** Initialize the MissingValueHandler.

**Parameters:**
- `df` (DataFrame): Input dataframe

**Example:**
```python
from auto_eda.missing_value_handler import MissingValueHandler
import pandas as pd

df = pd.DataFrame({
    'age': [25, None, 35],
    'income': [50000, 60000, None]
})

handler = MissingValueHandler(df)
```

---

### `analyze_missing_values()`

**Purpose:** Analyze missing values in the dataset.

**Returns:**
- `dict`: Dictionary with missing counts and percentages

**Example:**
```python
from auto_eda.missing_value_handler import MissingValueHandler
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age': [25, None, 35, None, 40],
    'income': [50000, None, 70000, 80000, None],
    'city': ['NYC', 'LA', None, 'Chicago', 'NYC']
})

handler = MissingValueHandler(df)
missing_info = handler.analyze_missing_values()

print("Missing counts:")
print(missing_info['counts'])

print("\nMissing percentages:")
print(missing_info['percentages'])
```

---

### `decide_strategy(column)`

**Purpose:** Decide the best strategy for handling missing values in a column.

**Parameters:**
- `column` (str): Column name

**Returns:**
- `str`: Strategy name ('mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill')

**Example:**
```python
from auto_eda.missing_value_handler import MissingValueHandler
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age': [25, 30, None, 35, 40],  # Normal distribution -> mean
    'income': [50000, 60000, None, 120000, 80000],  # Has outliers -> median
    'category': ['A', 'B', None, 'A', 'B'],  # Categorical -> mode
})

handler = MissingValueHandler(df)

# Check what strategy would be chosen
print(f"age strategy: {handler.decide_strategy('age')}")  # 'mean'
print(f"income strategy: {handler.decide_strategy('income')}")  # 'median'
print(f"category strategy: {handler.decide_strategy('category')}")  # 'mode'
```

---

### `handle_missing_values(auto=True)`

**Purpose:** Handle missing values based on intelligent decisions.

**Parameters:**
- `auto` (bool): If True, automatically decide strategies

**Returns:**
- `cleaned_df` (DataFrame): DataFrame with missing values handled
- `strategies` (dict): Dictionary of strategies applied

**Example:**
```python
from auto_eda.missing_value_handler import MissingValueHandler
import pandas as pd

df = pd.DataFrame({
    'age': [25, None, 35, None, 40],
    'income': [50000, None, 70000, 80000, None],
    'category': ['A', 'B', None, 'A', 'B']
})

handler = MissingValueHandler(df)
cleaned_df, strategies = handler.handle_missing_values(auto=True)

print("Strategies applied:", strategies)
print("\nCleaned dataframe:")
print(cleaned_df)
print(f"\nMissing values remaining: {cleaned_df.isnull().sum().sum()}")
```

---

### `get_summary()`

**Purpose:** Get summary of missing value handling.

**Returns:**
- `dict`: Summary dictionary

**Example:**
```python
from auto_eda.missing_value_handler import MissingValueHandler
import pandas as pd

df = pd.DataFrame({
    'age': [25, None, 35],
    'income': [50000, None, 70000]
})

handler = MissingValueHandler(df)
handler.handle_missing_values()

summary = handler.get_summary()
print("Summary:")
print(f"Total columns with missing: {summary['total_missing_columns']}")
print(f"Strategies: {summary['strategies_applied']}")
print(f"Missing info: {summary['missing_info']}")
```

---

## EDAAnalyzer Class

Performs comprehensive exploratory data analysis.

### `__init__(df)`

**Purpose:** Initialize the EDAAnalyzer.

**Parameters:**
- `df` (DataFrame): Input dataframe

**Example:**
```python
from auto_eda.eda_analyzer import EDAAnalyzer
import pandas as pd

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000]
})

analyzer = EDAAnalyzer(df)
print(f"Numeric columns: {analyzer.numeric_columns}")
print(f"Categorical columns: {analyzer.categorical_columns}")
```

---

### `get_basic_info()`

**Purpose:** Get basic information about the dataset.

**Returns:**
- `dict`: Dictionary with shape, columns, dtypes, memory usage, etc.

**Example:**
```python
from auto_eda.eda_analyzer import EDAAnalyzer
import pandas as pd

df = pd.DataFrame({
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

analyzer = EDAAnalyzer(df)
info = analyzer.get_basic_info()

print(f"Shape: {info['shape']}")
print(f"Columns: {info['columns']}")
print(f"Memory: {info['memory_usage']:.2f} MB")
print(f"Numeric: {info['numeric_columns']}")
print(f"Categorical: {info['categorical_columns']}")
```

---

### `get_descriptive_stats()`

**Purpose:** Get descriptive statistics for numeric and categorical columns.

**Returns:**
- `dict`: Dictionary with descriptive statistics

**Example:**
```python
from auto_eda.eda_analyzer import EDAAnalyzer
import pandas as pd

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA']
})

analyzer = EDAAnalyzer(df)
stats = analyzer.get_descriptive_stats()

print("Numeric stats:")
print(stats['numeric'])

print("\nCategorical stats:")
print(stats['categorical'])
```

---

### `detect_outliers(column, method='iqr')`

**Purpose:** Detect outliers in a numeric column.

**Parameters:**
- `column` (str): Column name
- `method` (str): 'iqr' or 'zscore'

**Returns:**
- `list`: List of outlier values

**Example:**
```python
from auto_eda.eda_analyzer import EDAAnalyzer
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 150, 200],  # 150 and 200 are outliers
    'income': [50000, 60000, 70000, 80000, 90000, 100000, 110000]
})

analyzer = EDAAnalyzer(df)

# Detect outliers using IQR method
outliers_iqr = analyzer.detect_outliers('age', method='iqr')
print(f"Outliers (IQR): {outliers_iqr}")

# Detect outliers using Z-score method
outliers_zscore = analyzer.detect_outliers('age', method='zscore')
print(f"Outliers (Z-score): {outliers_zscore}")
```

---

### `analyze_outliers()`

**Purpose:** Analyze outliers for all numeric columns.

**Returns:**
- `dict`: Dictionary with outlier analysis for each column

**Example:**
```python
from auto_eda.eda_analyzer import EDAAnalyzer
import pandas as pd

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 150],
    'income': [50000, 60000, 70000, 80000, 90000, 200000]
})

analyzer = EDAAnalyzer(df)
outliers = analyzer.analyze_outliers()

for col, info in outliers.items():
    print(f"{col}:")
    print(f"  Count: {info['count']}")
    print(f"  Percentage: {info['percentage']:.2f}%")
    print(f"  Sample outliers: {info['outliers'][:5]}")
```

---

### `calculate_correlations()`

**Purpose:** Calculate correlation matrix for numeric columns.

**Returns:**
- `dict`: Dictionary with correlation matrix and highly correlated pairs

**Example:**
```python
from auto_eda.eda_analyzer import EDAAnalyzer
import pandas as pd

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'experience': [1, 2, 3, 4, 5],  # Highly correlated with age
    'income': [50000, 60000, 70000, 80000, 90000]
})

analyzer = EDAAnalyzer(df)
correlations = analyzer.calculate_correlations()

print("Correlation matrix:")
print(correlations['matrix'])

print("\nHighly correlated pairs (|r| > 0.7):")
for pair in correlations['high_correlation_pairs']:
    print(f"{pair['column1']} <-> {pair['column2']}: {pair['correlation']:.3f}")
```

---

### `analyze_distributions()`

**Purpose:** Analyze distributions of numeric columns.

**Returns:**
- `dict`: Dictionary with distribution statistics for each column

**Example:**
```python
from auto_eda.eda_analyzer import EDAAnalyzer
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'normal_data': np.random.normal(100, 15, 1000),
    'skewed_data': np.random.exponential(2, 1000)
})

analyzer = EDAAnalyzer(df)
distributions = analyzer.analyze_distributions()

for col, dist_info in distributions.items():
    print(f"{col}:")
    print(f"  Mean: {dist_info['mean']:.2f}")
    print(f"  Median: {dist_info['median']:.2f}")
    print(f"  Skewness: {dist_info['skewness']:.3f}")
    print(f"  Kurtosis: {dist_info['kurtosis']:.3f}")
    print(f"  Is Normal: {dist_info['is_normal']}")
```

---

### `analyze_categorical()`

**Purpose:** Analyze categorical columns.

**Returns:**
- `dict`: Dictionary with categorical analysis for each column

**Example:**
```python
from auto_eda.eda_analyzer import EDAAnalyzer
import pandas as pd

df = pd.DataFrame({
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA', 'NYC'],
    'category': ['A', 'B', 'A', 'A', 'B', 'B']
})

analyzer = EDAAnalyzer(df)
categorical = analyzer.analyze_categorical()

for col, cat_info in categorical.items():
    print(f"{col}:")
    print(f"  Unique values: {cat_info['unique_count']}")
    print(f"  Most frequent: {cat_info['most_frequent']}")
    print(f"  Most frequent count: {cat_info['most_frequent_count']}")
    print(f"  Value counts: {cat_info['value_counts']}")
```

---

### `generate_visualizations(save_path=None)`

**Purpose:** Generate visualizations for the dataset.

**Parameters:**
- `save_path` (str, optional): Directory path to save plots

**Returns:**
- `list`: List of paths to saved visualization files

**Example:**
```python
from auto_eda.eda_analyzer import EDAAnalyzer
import pandas as pd
import os

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45] * 20,
    'income': [50000, 60000, 70000, 80000, 90000] * 20,
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'] * 20
})

analyzer = EDAAnalyzer(df)
os.makedirs('plots', exist_ok=True)

# Generate and save visualizations
viz_paths = analyzer.generate_visualizations(save_path='plots')

print("Saved visualizations:")
for path in viz_paths:
    print(f"  - {path}")
# Output:
#   - plots/distributions.png
#   - plots/correlation_heatmap.png
#   - plots/boxplots.png
#   - plots/categorical_counts.png
```

---

### `run_full_analysis(save_plots=False, plot_path=None)`

**Purpose:** Run complete EDA analysis (all methods in sequence).

**Parameters:**
- `save_plots` (bool): Whether to save plots
- `plot_path` (str, optional): Directory to save plots

**Returns:**
- `dict`: Dictionary with all analysis results

**Example:**
```python
from auto_eda.eda_analyzer import EDAAnalyzer
import pandas as pd

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA']
})

analyzer = EDAAnalyzer(df)

# Run all analysis
results = analyzer.run_full_analysis(save_plots=True, plot_path='output')

# Access all results
print("Available results:", results.keys())
print("Basic info:", results['basic_info'])
print("Outliers:", results['outliers'])
print("Correlations:", results['correlations'])
```

---

## ReportGenerator Class

Generates comprehensive EDA reports.

### `__init__(df, missing_handler_summary, eda_results, strategies_applied)`

**Purpose:** Initialize the ReportGenerator.

**Parameters:**
- `df` (DataFrame): Cleaned dataframe
- `missing_handler_summary` (dict): Summary from MissingValueHandler
- `eda_results` (dict): Results from EDAAnalyzer
- `strategies_applied` (dict): Strategies applied for missing values

**Example:**
```python
from auto_eda.report_generator import ReportGenerator
from auto_eda import AutoEDA
import pandas as pd

df = pd.DataFrame({
    'age': [25, None, 35],
    'income': [50000, None, 70000]
})

eda = AutoEDA(df=df)
eda.handle_missing_values()
eda.perform_eda()

# Create report generator
report_gen = ReportGenerator(
    df=eda.df,
    missing_handler_summary=eda.missing_handler.get_summary(),
    eda_results=eda.results,
    strategies_applied=eda.strategies_applied
)
```

---

### `generate_text_report()`

**Purpose:** Generate a comprehensive text report.

**Returns:**
- `str`: The generated report as a string

**Example:**
```python
from auto_eda.report_generator import ReportGenerator
from auto_eda import AutoEDA
import pandas as pd

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000]
})

eda = AutoEDA(df=df)
eda.perform_eda()

report_gen = ReportGenerator(
    df=eda.df,
    missing_handler_summary={},
    eda_results=eda.results,
    strategies_applied={}
)

# Generate report
report = report_gen.generate_text_report()
print(report)
```

---

### `save_report(filepath)`

**Purpose:** Save the report to a file.

**Parameters:**
- `filepath` (str): Path to save the report

**Returns:**
- `str`: Path to saved file

**Example:**
```python
from auto_eda.report_generator import ReportGenerator
from auto_eda import AutoEDA
import pandas as pd

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000]
})

eda = AutoEDA(df=df)
eda.perform_eda()

report_gen = ReportGenerator(
    df=eda.df,
    missing_handler_summary={},
    eda_results=eda.results,
    strategies_applied={}
)

# Save report
saved_path = report_gen.save_report('my_report.txt')
print(f"Report saved to: {saved_path}")
```

---

## Complete Workflow Example

Here's a complete example using all the main functions:

```python
import pandas as pd
from auto_eda import AutoEDA

# 1. Load or create your dataset
df = pd.read_csv('your_data.csv')
# OR
df = pd.DataFrame({
    'age': [25, 30, None, 35, 40],
    'income': [50000, None, 70000, 80000, 90000],
    'category': ['A', 'B', None, 'A', 'B'],
    'target': [0, 1, 0, 1, 0]
})

# 2. Initialize AutoEDA
eda = AutoEDA(df=df)

# 3. Run complete workflow
results = eda.run_complete_eda(
    auto_handle_missing=True,  # Auto-handle missing values
    save_plots=True,            # Save visualizations
    save_report=True,           # Save report
    get_split=True,             # Do train-test split
    target_column='target'      # Specify target column
)

# 4. Access results
cleaned_df = results['cleaned_df']
original_df = results['original_df']
eda_results = results['results']
strategies = results['strategies_applied']

# 5. Get train-test split
if results['split_data']:
    X_train, X_test, y_train, y_test = results['split_data']
    
    # Now you're ready to build models!
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
```

---

## Summary

This package provides a complete automatic EDA solution:

1. **AutoEDA** - Main class that orchestrates everything
2. **MissingValueHandler** - Intelligently handles missing values
3. **EDAAnalyzer** - Performs comprehensive data analysis
4. **ReportGenerator** - Creates detailed reports

All functions work together to provide a seamless EDA experience, from loading data to preparing it for model building!

