# Advanced Text Processing Library

[english](./README-en.md), [中文](./README.md).

## Overview

This library encompasses a series of Python classes and functions designed for advanced text processing, including model configuration, byte pair encoding (BPE), regex-based tokenization, and control character handling. These components are suitable for constructing and operating on natural language processing (NLP) models, especially those that require sophisticated text manipulation and configuration management.

## Components

### 1. Config Class

**File**: `config.py`

- **Description**: Loads configuration parameters for NLP models from a JSON file.
- **Usage**: Initialize the model with predefined parameters such as dimensions, layer count, and vocabulary size.

### 2. Llama3 Model

**File**: `model.py`

- **Description**: A neural network model using the PaddlePaddle framework, incorporating embedding layers, RMS normalization, and attention mechanisms.
- **Features**:
  - Embedding layer for input token handling.
  - Customizable attention blocks for model depth and complexity.
  - Forward pass definition for neural processing.

### 3. CoreBPE and RegexTokenizer

**Files**: `core_bpe.py`, `./bpe/regex_tokenizer.py`

- **Description**: Implements byte pair encoding and regex-based tokenization for efficient text preprocessing.
- **Features**:
  - `CoreBPE` handles merging of byte pairs and special token recognition.
  - `RegexTokenizer` utilizes regular expressions to split text, suitable for parsing and tokenizing textual input according to patterns observed in GPT models.

### 4. Utility Functions

**File**: `bpe/base.py`

- **Description**: Includes various utility functions such as `get_stats` for frequency counting of byte pairs, `merge` for combining identified byte pairs, and character replacement functions to handle control characters in texts.
- **Usage**: Directly invoked within the tokenization and encoding processes to manage text data.

## Installation

Ensure you have Python 3.6+ installed, and then install the required packages:

```bash
pip install paddlepaddle
pip install regex
```

## Usage

### Configuring Models

Load configurations using the `Config` class:

```python
from config import Config
config = Config()
```

### Encoding and Decoding Text

Utilize the `CoreBPE` or `RegexTokenizer` classes to encode or decode text:

```python
from core_bpe import CoreBPE
bpe = CoreBPE(mergeable_ranks, special_tokens, pattern)
encoded_text = bpe.encode("Example text")
decoded_text = bpe.decode(encoded_text)

from regex_tokenizer import RegexTokenizer
tokenizer = RegexTokenizer()
encoded_text = tokenizer.encode("Example text")
decoded_text = tokenizer.decode(encoded_text)
```

### Training Tokenizers

Train a tokenizer based on new text corpora:

```python
tokenizer.train("New corpus text", vocab_size=1000, verbose=True)
```

## Contributing

Contributions to this library are welcome. Please ensure to maintain the existing coding style, add unit tests for any new or changed functionality, and update the docs as needed.
