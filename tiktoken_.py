from typing import AbstractSet, Collection, Literal, Union
from test import CoreBPE
import functools

class Encoding:
    def __init__(
        self,
        pat_str: str,
        mergeable_ranks: dict[bytes, int],
        special_tokens: dict[str, int],
    ):
        """
        初始化编码器类。
        Args:
            pat_str (str): 正则表达式，用于文本分割。
            mergeable_ranks (dict[bytes, int]): 可合并的标记及其优先级。
            special_tokens (dict[str, int]): 特殊标记及其对应的ID。
        """
        self._pat_str = pat_str
        self._mergeable_ranks = mergeable_ranks
        self._special_tokens = special_tokens

        # 创建核心BPE编码对象，用于处理编码和解码操作。
        self._core_bpe = CoreBPE(mergeable_ranks, special_tokens, pat_str)

    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # noqa: B006
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> list[int]:
        """
        对给定文本进行编码。
        Args:
            text (str): 输入的文本。
            allowed_special (Union[Literal["all"], AbstractSet[str]]): 允许的特殊标记集合。
            disallowed_special (Union[Literal["all"], Collection[str]]): 不允许的特殊标记集合。
        Returns:
            list[int]: 编码后的标记ID列表。
        """
        if disallowed_special == "all":
            # 如果禁用所有特殊标记，则从所有特殊标记中减去允许的特殊标记集合
            disallowed_special = self.special_tokens_set - allowed_special

        if isinstance(allowed_special, frozenset):
            allowed_special = set(allowed_special)

        # 尝试编码文本，如果遇到Unicode编码错误，使用utf-16编码再尝试一次
        try:
            return self._core_bpe.encode(text, allowed_special)
        except UnicodeEncodeError:
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
            return self._core_bpe.encode(text, allowed_special)

    def decode(self, tokens: list[int], errors: str = "replace") -> str:
        """
        解码给定的标记ID列表。
        Args:
            tokens (list[int]): 标记ID列表。
            errors (str): 解码错误的处理策略。
        Returns:
            str: 解码后的文本。
        """
        return self._core_bpe.decode_bytes(tokens).decode("utf-8", errors=errors)

    @functools.cached_property
    def special_tokens_set(self) -> set[str]:
        """
        返回一个包含所有特殊标记的集合。
        Returns:
            set[str]: 特殊标记的集合。
        """
        return set(self._special_tokens.keys())
