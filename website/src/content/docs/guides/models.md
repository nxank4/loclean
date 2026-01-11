---
title: Model Management
description: Download and manage GGUF models for local inference.
---

## Overview

Loclean automatically downloads models on first use, but you can pre-download them using the CLI for faster startup times.

## CLI Commands

### Download a Model

```bash
loclean model download --name phi-3-mini
```

### List Available Models

```bash
loclean model list
```

This shows all available models with their sizes and descriptions.

### Check Download Status

```bash
loclean model status
```

This shows which models are already downloaded and their cache locations.

## Available Models

- **phi-3-mini**: Microsoft Phi-3 Mini (3.8B, 4K context) - Default, balanced
- **tinyllama**: TinyLlama 1.1B - Smallest, fastest
- **gemma-2b**: Google Gemma 2B Instruct - Balanced performance
- **qwen3-4b**: Qwen3 4B - Higher quality
- **gemma-3-4b**: Gemma 3 4B - Larger context
- **deepseek-r1**: DeepSeek R1 - Reasoning model

## Model Cache

Models are cached in `~/.cache/loclean` by default. You can specify a custom cache directory using the `--cache-dir` option:

```bash
loclean model download --name phi-3-mini --cache-dir /path/to/cache
```

## Choosing a Model

- **For Speed**: Use `tinyllama` (smallest, fastest)
- **For Balance**: Use `phi-3-mini` (default, good performance)
- **For Quality**: Use `qwen3-4b` or `gemma-3-4b` (larger, better quality)
- **For Reasoning**: Use `deepseek-r1` (specialized for reasoning tasks)
