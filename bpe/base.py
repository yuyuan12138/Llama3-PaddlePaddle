import unicodedata

def get_stats(ids: list[int], counts: dict[tuple, int] = None) -> dict[tuple, int]:
    """ 统计 id 序列中相邻元素对出现的频率。
    Args:
        ids: 由整数组成的列表。
        counts: 存储元素对频率的字典。
    Returns:
        更新后的频率字典。
    """
    if counts is None:
        counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids: list[int], pair: tuple, idx: int) -> list[int]:
    """ 在 id 列表中合并指定的元素对。
    Args:
        ids: 整数列表。
        pair: 需要合并的元素对。
        idx: 合并后的新元素索引。
    Returns:
        合并后的新列表。
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2  # 跳过已合并的一对
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def replace_control_characters(s: str) -> str:
    """ 替换字符串中的控制字符。
    Args:
        s: 输入的字符串。
    Returns:
        替换控制字符后的字符串。
    """
    return "".join(ch if unicodedata.category(ch)[0] != "C" else f"\\u{ord(ch):04x}" for ch in s)

def render_token(t: bytes) -> str:
    """ 将字节序列解码为 UTF-8 字符串，并替换控制字符。
    Args:
        t: 字节序列。
    Returns:
        解码和处理后的字符串。
    """
    s = t.decode('utf-8', errors='replace')
    return replace_control_characters(s)

class Tokenizer:
    """ 一个简单的字节对编码器 (BPE) 分词器。"""

    def __init__(self):
        # 初始化参数，无合并操作，仅基于字节
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        """ 构建词汇表，基于合并规则和特殊字符。"""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """ 保存模型和词汇表到文件。"""
        model_file = f"{file_prefix}.model"
        vocab_file = f"{file_prefix}.vocab"
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for (idx1, idx2), idx in self.merges.items():
                f.write(f"{idx1} {idx2} {idx}\n")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """ 从文件加载模型。"""
        assert model_file.endswith(".model")
        with open(model_file, 'r', encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "minbpe v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            self.special_tokens = {f.readline().strip().split()[0]: int(s) for _ in range(num_special)}
            self.merges = {(int(line.split()[0]), int(line.split()[1])): int(line.split()[2]) for line in f}
        self.vocab = self._build_vocab()
