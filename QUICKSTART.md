# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Basic Usage

### Simplest Example

```python
from auto_eda import AutoEDA
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Create AutoEDA instance
eda = AutoEDA(df=df)

# Run everything automatically
results = eda.run_complete_eda()
```

That's it! The library will:
1. ✅ Handle all missing values automatically
2. ✅ Perform comprehensive EDA analysis
3. ✅ Generate visualizations
4. ✅ Create detailed reports
5. ✅ Ask you questions when needed (output directory, train-test split, etc.)

### What You Get

After running `run_complete_eda()`, you'll have:

- **Cleaned Dataset**: `results['cleaned_df']` - Ready for modeling
- **EDA Results**: `results['results']` - All analysis findings
- **Strategies Applied**: `results['strategies_applied']` - How missing values were handled
- **Visualizations**: Saved in the output directory (if you chose to save)
- **Report**: Text file with complete summary

### With Train-Test Split

```python
# Run with train-test split
results = eda.run_complete_eda(get_split=True)

# Or get split separately
X_train, X_test, y_train, y_test = eda.get_train_test_split(target_column='target')
```

### From File Path

```python
eda = AutoEDA(file_path='data.csv')
results = eda.run_complete_eda()
```

## Next Steps

After EDA is complete, your dataset is ready for:
- Feature engineering
- Model building
- Training machine learning models

The library has done all the preprocessing and analysis for you!

