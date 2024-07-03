# 高级文本处理库

## 概述

本库包含一系列为高级文本处理设计的 Python 类和函数，包括模型配置、字节对编码（BPE）、基于正则表达式的分词和控制字符处理。这些组件适用于构建和操作自然语言处理（NLP）模型，特别是那些需要复杂文本操作和配置管理的模型。

## 组件

### 1. 配置类

**文件**：`config.py`

- **描述**：从 JSON 文件加载 NLP 模型的配置参数。
- **用法**：使用预定义的参数（如维度、层数和词汇表大小）初始化模型。

### 2. Llama3 模型

**文件**：`model.py`

- **描述**：使用 PaddlePaddle 框架的神经网络模型，包括嵌入层、RMS 归一化和注意力机制。
- **特点**：
  - 输入令牌处理的嵌入层。
  - 可定制的注意力模块，增加模型的深度和复杂性。
  - 定义神经处理的前向传播。

### 3. CoreBPE 和 RegexTokenizer

**文件**：`core_bpe.py`, `./bpe/regex_tokenizer.py`

- **描述**：实现字节对编码和基于正则表达式的分词，用于高效的文本预处理。
- **特点**：
  - `CoreBPE` 负责字节对的合并和特殊标记的识别。
  - `RegexTokenizer` 利用正则表达式分割文本，适合按照 GPT 模型观察到的模式解析和标记化文本输入。

### 4. 实用功能

**文件**：`bpe/base.py`

- **描述**：包括各种实用功能，如 `get_stats` 用于计数字节对的频率，`merge` 用于组合已识别的字节对，以及处理文本中控制字符的字符替换功能。
- **用法**：在分词和编码过程中直接调用，以管理文本数据。

## 安装

确保安装了 Python 3.6 及以上版本，然后安装所需的包：

```bash
pip install paddlepaddle
pip install regex
```

## 使用方法

### 配置模型

使用 `Config` 类加载配置：

```python
from config import Config
config = Config()
```

### 编码和解码文本

使用 `CoreBPE` 或 `RegexTokenizer` 类来编码或解码文本：

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

### 训练分词器

基于新的文本语料库训练分词器：

```python
tokenizer.train("New corpus text", vocab_size=1000, verbose=True)
```

## 贡献

欢迎对本库做出贡献。请确保保持现有的编码风格，为任何新功能或变更添加单元测试，并根据需要更新文档。