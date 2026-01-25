# Loclean Examples

This directory contains interactive Jupyter notebooks demonstrating Loclean's features.

## Recommended: Use Jupyter notebooks

We **strongly recommend** using Jupyter notebooks to run these examples:

- **Interactive**: Run cells individually and see results immediately
- **Explorable**: Modify code and experiment with different inputs
- **Educational**: See outputs, errors, and intermediate results
- **Shareable**: Easy to share with others

## Getting Started

### Option 1: Jupyter notebook (recommended)

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
- Data cleaning with dataframes
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

For more advanced features like model selection, caching strategies, and performance optimization, check out the [full documentation](https://nxank4.github.io/loclean).

### 4. [04-structured-extraction.ipynb](./04-structured-extraction.ipynb)
Advanced structured extraction:
- Complex nested schemas
- Union types
- Error handling and retries
- Performance optimization
- Caching demonstrations

### 5. [05-debug-mode.ipynb](./05-debug-mode.ipynb)
Debugging and detailed logging:
- Enabling verbose mode
- Seeing raw LLM prompts and outputs
- Debugging Pydantic validation issues
- Global configuration via environment variables

## Directory Structure

This directory contains:

- **`*.ipynb`**: Jupyter notebook files demonstrating specific features. Numbered prefixes indicate recommended reading order.
- **`benchmark.py`**: Performance benchmarking script for comparing different models and configurations. Run with:
  ```bash
  python examples/benchmark.py
  ```
- **`README.md`**: This file - documentation and guidelines for examples.

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
- **Model download**: First run will download the model (one-time, ~2GB). Loclean supports multiple models including Phi-3, Qwen, Gemma, DeepSeek, TinyLlama, and LFM2.5. See the [main README](../README.md#available-models) for the full list.
- **Caching**: Results are cached, so re-running cells is fast
- **Errors?** Check that you have the required dependencies installed

## Documentation

- **Full Documentation**: [https://nxank4.github.io/loclean](https://nxank4.github.io/loclean)
- **GitHub Repository**: [https://github.com/nxank4/loclean](https://github.com/nxank4/loclean)
- **PyPI Package**: [https://pypi.org/project/loclean](https://pypi.org/project/loclean)

## Contributing

Found a bug or want to add an example? Please open an issue or pull request on GitHub!

### Guidelines for Contributors

When adding a new example notebook:

1. **Naming convention**: Use numbered prefixes (e.g., `06-new-feature.ipynb`) to maintain order
2. **Structure**: Follow the pattern of existing notebooks:
   - Start with a clear title and description
   - Include installation/setup cells
   - Provide clear explanations in markdown cells
   - Show expected outputs
3. **Dependencies**: Document any special dependencies in the notebook's first cell
4. **Testing**: Ensure all cells run successfully before submitting
5. **Documentation**: Update this README to include your new notebook in the "Available Notebooks" section

### Code Style Guidelines

- **Keep examples simple**: Focus on demonstrating one feature or concept per notebook
- **Use real-world scenarios**: Make examples relatable and practical
- **Document assumptions**: Clearly state any prerequisites or assumptions
- **Test thoroughly**: Ensure all cells execute without errors
- **Follow code style**: Use type hints and follow PEP 8 (enforced by `ruff`)
- **Update this README**: When adding new notebooks, update the "Available Notebooks" section above

### Benchmark Script

The `benchmark.py` script is used for performance testing. When modifying it:

- Keep it focused on performance metrics
- Document what is being benchmarked
- Ensure it runs without errors
- Update this README if the script's purpose changes significantly
