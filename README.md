# Auto EDA - Automatic Exploratory Data Analysis Library

A completely automatic Python library for performing comprehensive Exploratory Data Analysis (EDA) on any dataset. Just import, load your data, and run - the library handles everything from missing values to generating detailed reports.

## Features

✨ **Fully Automatic**: Just provide your dataset and the library does the rest  
🔍 **Intelligent Missing Value Handling**: Automatically decides whether to use mean, median, mode, drop, or fill strategies  
📊 **Comprehensive EDA**: Statistics, distributions, correlations, outliers, and more  
📈 **Visualizations**: Automatic generation of plots and charts  
📝 **Detailed Reports**: Complete summary of all analysis performed  
💬 **Interactive**: Asks user questions when input is needed (train-test split, etc.)  
🎯 **Model-Ready**: Prepares your dataset for machine learning model building

## Installation

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## Quick Start

### Basic Usage

```python
from auto_eda import AutoEDA
import pandas as pd

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Initialize AutoEDA
eda = AutoEDA(df=df)

# Run complete EDA workflow
results = eda.run_complete_eda()
```

### From File Path

```python
from auto_eda import AutoEDA

# Initialize with file path
eda = AutoEDA(file_path='your_dataset.csv')

# Run complete EDA workflow
results = eda.run_complete_eda()
```

### Step-by-Step Usage

```python
from auto_eda import AutoEDA
import pandas as pd

# Load dataset
df = pd.read_csv('your_dataset.csv')

# Initialize
eda = AutoEDA(df=df)

# Step 1: Handle missing values (automatic)
eda.handle_missing_values(auto=True)

# Step 2: Perform EDA analysis
eda.perform_eda(save_plots=True)

# Step 3: Generate report
eda.generate_report(save_to_file=True)

# Step 4: Get train-test split (interactive)
X_train, X_test, y_train, y_test = eda.get_train_test_split()
```

## What Auto EDA Does

### 1. Missing Value Handling
- **Automatic Detection**: Identifies all missing values in the dataset
- **Intelligent Strategy Selection**:
  - **Numeric columns**: Uses mean for normal distributions, median for skewed data or outliers
  - **Categorical columns**: Uses mode (most frequent value)
  - **High missing percentage (>50%)**: Drops the column
  - **Time series**: Uses forward fill
- **User Control**: Can be set to ask user for each column's strategy

### 2. Comprehensive EDA Analysis
- **Basic Information**: Dataset shape, memory usage, column types
- **Descriptive Statistics**: Mean, median, std, min, max for all numeric columns
- **Distribution Analysis**: Skewness, kurtosis, normality tests
- **Outlier Detection**: IQR and Z-score methods
- **Correlation Analysis**: Correlation matrix and highly correlated pairs
- **Categorical Analysis**: Value counts, unique values, most frequent categories

### 3. Visualizations
- Distribution plots for numeric columns
- Correlation heatmaps
- Box plots for outlier visualization
- Categorical value count charts

### 4. Report Generation
- Comprehensive text report with all findings
- Recommendations for next steps
- Summary of all transformations applied
- Saved to file (optional)

### 5. Train-Test Split
- Interactive prompts for target column selection
- Configurable test size
- Random state for reproducibility
- Returns ready-to-use train and test sets

## Example Output

When you run `run_complete_eda()`, you'll see:

```
================================================================================
AUTOMATIC EDA WORKFLOW
================================================================================
Dataset shape: (1000, 10)
Columns: 10

Do you want to save EDA results to a directory? (yes/no) (default: yes): yes
Enter output directory path (press Enter for 'eda_output'): 
Results will be saved to: eda_output

================================================================================
STEP 1: HANDLING MISSING VALUES
================================================================================

Found missing values in 3 columns:
  - age: 50 missing (5.00%)
  - income: 20 missing (2.00%)
  - category: 10 missing (1.00%)

Automatically deciding strategies for handling missing values...

Strategies applied:
  - age: median
  - income: mean
  - category: mode

✓ Missing values handled successfully!

================================================================================
STEP 2: PERFORMING EXPLORATORY DATA ANALYSIS
================================================================================
Running comprehensive EDA analysis...
  - Gathering basic information...
  - Calculating descriptive statistics...
  - Analyzing distributions...
  - Detecting outliers...
  - Calculating correlations...
  - Analyzing categorical variables...
  - Generating visualizations...
EDA analysis complete!

================================================================================
STEP 3: GENERATING REPORT
================================================================================

================================================================================
AUTOMATIC EDA REPORT
================================================================================
...
[Detailed report with all findings]
...

================================================================================
EDA WORKFLOW COMPLETE!
================================================================================

Your dataset is now ready for model building!
Cleaned dataset shape: (1000, 10)
Original dataset shape: (1000, 10)
```

## API Reference

### AutoEDA Class

#### `__init__(df=None, file_path=None)`
Initialize AutoEDA with a dataframe or file path.

#### `run_complete_eda(auto_handle_missing=True, save_plots=True, save_report=True, get_split=False, target_column=None)`
Run the complete EDA workflow.

**Parameters:**
- `auto_handle_missing` (bool): Automatically handle missing values
- `save_plots` (bool): Save visualization plots
- `save_report` (bool): Save report to file
- `get_split` (bool): Perform train-test split
- `target_column` (str): Target column for split

**Returns:** Dictionary with cleaned dataframe, results, and split data

#### `handle_missing_values(auto=True)`
Handle missing values in the dataset.

#### `perform_eda(save_plots=True)`
Perform comprehensive EDA analysis.

#### `generate_report(save_to_file=True)`
Generate and save EDA report.

#### `get_train_test_split(target_column=None, test_size=None, random_state=None)`
Get train-test split with interactive prompts.

#### `get_cleaned_dataframe()`
Get the cleaned dataframe after EDA.

#### `get_results()`
Get all EDA results.

## Requirements

- Python >= 3.7
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0

## Project Structure

```
datascience_package/
├── auto_eda/
│   ├── __init__.py
│   ├── auto_eda.py          # Main AutoEDA class
│   ├── missing_value_handler.py  # Missing value handling logic
│   ├── eda_analyzer.py       # EDA analysis functions
│   └── report_generator.py   # Report generation
├── setup.py
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Author

Your Name

