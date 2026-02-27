# Loclean Examples

This directory contains interactive Jupyter notebooks demonstrating Loclean's features.

## Recommended: Use Jupyter notebooks

We **strongly recommend** using Jupyter notebooks to run these examples:

- **Interactive**: Run cells individually and see results immediately
- **Explorable**: Modify code and experiment with different inputs
- **Educational**: See outputs, errors, and intermediate results
- **Shareable**: Easy to share with others

## Getting Started

### Prerequisites

Loclean uses **Ollama** for local inference. Install it once:

```bash
# Linux / WSL
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# All platforms → https://ollama.com/download
```

> **Note:** You do not need to manually start the daemon or pull models.
> Loclean handles both automatically on first use:
>
> 1. **Auto-start** — if the Ollama daemon is not running, Loclean launches it for you.
> 2. **Auto-pull** — if the requested model is missing, Loclean downloads it with a progress bar.

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

### 6. [06-entity-resolution.ipynb](./06-entity-resolution.ipynb)
Entity resolution — canonicalize messy string variations:
- Merge company-name typos, abbreviations, casing
- Configurable similarity threshold
- Before/after comparison

### 7. [07-oversampling.ipynb](./07-oversampling.ipynb)
Semantic oversampling for imbalanced datasets:
- Pydantic-schema-driven synthetic record generation
- Minority-class augmentation
- Class distribution balancing

### 8. [08-log-shredding.ipynb](./08-log-shredding.ipynb)
Log shredding — parse unstructured logs into relational tables:
- Mixed log format parsing (auth, API, payment, inventory, ML)
- Automatic schema inference
- One column → multiple normalized DataFrames

### 9. [09-feature-discovery.ipynb](./09-feature-discovery.ipynb)
Automated feature discovery:
- LLM-proposed mathematical transformations
- Housing price dataset example
- Mutual information maximisation with target variable

### 10. [10-quality-validation.ipynb](./10-quality-validation.ipynb)
Data quality validation with natural-language rules:
- Plain-English constraint definitions
- Structured compliance reports
- Multi-rule evaluation

### 11. [11-kaggle-housing-pipeline.ipynb](./11-kaggle-housing-pipeline.ipynb)
🏠 **Data Science** — Kaggle-style housing prediction workflow:
- Clean messy strings → entity resolution → feature discovery
- Minority-class oversampling → quality validation → PII scrubbing
- Full pipeline with `qwen2.5-coder:1.5b`

### 12. [12-log-engineering-pipeline.ipynb](./12-log-engineering-pipeline.ipynb)
🔧 **Data Engineering** — log processing and warehouse loading:
- Structured extraction with Pydantic schemas
- Compiled extraction for high-performance parsing
- Log shredding into relational tables → quality gates → PII masking

### 13. [13-trap-pruning.ipynb](./13-trap-pruning.ipynb)
Trap feature detection and removal:
- Statistical profiling of numeric columns
- LLM-verified Gaussian noise detection
- Before/after column comparison with verdicts

### 14. [14-missingness-recognition.ipynb](./14-missingness-recognition.ipynb)
Missing Not At Random (MNAR) pattern detection:
- Detect informative missingness patterns
- Automatic boolean feature flag encoding
- Clinical dataset example (income ↔ employment)

### 15. [15-leakage-auditing.ipynb](./15-leakage-auditing.ipynb)
Target leakage detection and removal:
- Semantic timeline evaluation per column
- Domain-aware reasoning (loan approval example)
- Automatic removal of leaked features

### 16. [16-instruction-optimization.ipynb](./16-instruction-optimization.ipynb)
Reward-driven prompt optimization:
- Generates structural instruction variations
- Scores each against validation sample (field-level F1)
- Returns the best-performing extraction instruction

## Standalone Scripts

| Script | Description |
|--------|-------------|
| [`benchmark.py`](./benchmark.py) | Performance benchmark: vectorized dedup + cache speedup on 100K rows |
| [`eval_demo.py`](./eval_demo.py) | Evaluation framework demo with optional Langfuse tracking |

## Directory Structure

This directory contains:

- **`*.ipynb`**: Jupyter notebooks demonstrating specific features. Numbered prefixes indicate recommended reading order.
- **`benchmark.py`**: Performance benchmarking script.
- **`eval_demo.py`**: Evaluation framework demo.
- **`README.md`**: This file.

## Requirements

```bash
# Install Loclean (Ollama daemon is started automatically)
pip install loclean

# For privacy scrubbing with fake data replacement
pip install loclean[privacy]

# For Jupyter notebooks
pip install jupyter

# Optional: For better performance
pip install polars pandas
```

## Model Management

Loclean auto-pulls models on first use. You can also manage models explicitly via the CLI:

```bash
# Check daemon status and list local models
loclean model status

# Pull a specific model ahead of time
loclean model pull phi3
loclean model pull llama3
```

## Running Examples

1. **Start Jupyter**: `jupyter notebook` or `jupyter lab`
2. **Open a notebook**: Click on any `.ipynb` file
3. **Run cells**: Press `Shift+Enter` to run a cell
4. **Experiment**: Modify code and see results

## Tips

- **First time?** Start with `01-quick-start.ipynb`
- **Need help?** Check the [full documentation](https://nxank4.github.io/loclean)
- **Model auto-pull**: First run auto-downloads the default model (one-time, ~2 GB). Change models with `loclean.clean(..., model="llama3")` or set `LOCLEAN_MODEL=llama3`.
- **Caching**: Results are cached, so re-running cells is fast
- **Errors?** Check that you have Ollama installed (`ollama --version`) and the required Python dependencies

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
