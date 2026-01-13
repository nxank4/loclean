# Loclean Examples

This directory contains interactive Jupyter notebooks demonstrating Loclean's features.

## 📓 Recommended: Use Jupyter Notebooks

We **strongly recommend** using Jupyter notebooks to run these examples:

- **Interactive**: Run cells individually and see results immediately
- **Explorable**: Modify code and experiment with different inputs
- **Educational**: See outputs, errors, and intermediate results
- **Shareable**: Easy to share with others

## Getting Started

### Option 1: Jupyter Notebook (Recommended)

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```

Then open any `.ipynb` file in the browser.

### Option 2: VS Code

VS Code has built-in Jupyter notebook support. Just open any `.ipynb` file.

### Option 3: Google Colab

Upload any `.ipynb` file to [Google Colab](https://colab.research.google.com/) and run it there.

## Available Notebooks

### 1. [01-quick-start.ipynb](./01-quick-start.ipynb)
**Start here!** Core features and basic usage:
- Structured extraction with Pydantic
- Data cleaning with DataFrames
- Privacy scrubbing
- Working with Pandas/Polars

### 2. [02-data-cleaning.ipynb](./02-data-cleaning.ipynb)
Comprehensive data cleaning examples:
- Basic usage and custom instructions
- Working with different backends
- Batch and parallel processing
- Handling missing values
- Model selection

### 3. [03-privacy-scrubbing.ipynb](./03-privacy-scrubbing.ipynb)
Privacy-first PII scrubbing:
- Mask and replace modes
- Selective scrubbing strategies
- Locale support
- Before/after examples

### 4. [04-structured-extraction.ipynb](./04-structured-extraction.ipynb)
Advanced structured extraction:
- Complex nested schemas
- Union types
- Error handling and retries
- Performance optimization
- Caching demonstrations


## Requirements

```bash
# Install Loclean
pip install loclean

# For privacy scrubbing with fake data replacement
pip install loclean[privacy]

# For Jupyter notebooks
pip install jupyter

# Optional: For better performance
pip install polars pandas
```

## Running Examples

1. **Start Jupyter**: `jupyter notebook` or `jupyter lab`
2. **Open a notebook**: Click on any `.ipynb` file
3. **Run cells**: Press `Shift+Enter` to run a cell
4. **Experiment**: Modify code and see results

## Tips

- **First time?** Start with `01-quick-start.ipynb`
- **Need help?** Check the [full documentation](https://nxank4.github.io/loclean)
- **Model download**: First run will download the model (one-time, ~2GB)
- **Caching**: Results are cached, so re-running cells is fast
- **Errors?** Check that you have the required dependencies installed

## Documentation

- **Full Documentation**: [https://nxank4.github.io/loclean](https://nxank4.github.io/loclean)
- **GitHub Repository**: [https://github.com/nxank4/loclean](https://github.com/nxank4/loclean)
- **PyPI Package**: [https://pypi.org/project/loclean](https://pypi.org/project/loclean)

## Contributing

Found a bug or want to add an example? Please open an issue or pull request on GitHub!
