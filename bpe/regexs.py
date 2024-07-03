import regex as re
from .base import Tokenizer, get_stats, merge

# GPT-2 和 GPT-4 使用的文本分割正则表达式
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern: str = None):
        """
        初始化一个正则表达式分词器。
        Args:
            pattern (str): 使用的分割模式，默认为 GPT-4 的模式。
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """
        从文本中训练词汇表。
        Args:
            text (str): 输入的文本。
            vocab_size (int): 目标词汇表的大小，必须不小于256。
            verbose (bool): 是否打印详细的合并过程信息。
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)  # 使用正则表达式分割文本
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]  # 将文本块编码为字节

        merges = {}  # 合并操作记录
        vocab = {idx: bytes([idx]) for idx in range(256)}  # 初始化词汇表
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)  # 统计每对相邻字节的出现次数
            pair = max(stats, key=stats.get)  # 选择出现最频繁的字节对
            idx = 256 + i

            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]  # 更新 id 序列
            merges[pair] = idx  # 记录合并操作
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]  # 更新词汇表

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens: dict):
        """
        注册特殊标记。
        Args:
            special_tokens (dict): 特殊标记的字典，格式为 {str: int}。
        """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        """
        将 token id 列表解码为字符串。
        Args:
            ids (list[int]): token id 列表。
        Returns:
            str: 解码后的字符串。
        """
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode_ordinary(self, text):
        """
        仅对文本进行普通编码，不处理特殊标记。
        Args:
            text (str): 输入的文本。
        Returns:
            list[int]: 编码后的 token id 列表。
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        编码文本，处理特殊标记。
        Args:
            text (str): 输入的文本。
            allowed_special (str): 特殊标记的处理方式。
        Returns:
            list[int]: 编码后的 token id 列表。
        """
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
