from typing import AbstractSet, Collection, Literal, Union
from tiktoken_cpy import _tiktoken
import functools

class Encoding:
    def __init__(
        self,
        pat_str: str,
        mergeable_ranks: dict[bytes, int],
        special_tokens: dict[str, int],
    ):

        self._pat_str = pat_str
        self._mergeable_ranks = mergeable_ranks
        self._special_tokens = special_tokens

        self._core_bpe = _tiktoken.CoreBPE(mergeable_ranks, special_tokens, pat_str)

    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # noqa: B006
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> list[int]:
        if disallowed_special == "all":
            disallowed_special = self.special_tokens_set - allowed_special

        if isinstance(allowed_special, frozenset):
            allowed_special = set(allowed_special)

        try:
            return self._core_bpe.encode(text, allowed_special)
        except UnicodeEncodeError:
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
            return self._core_bpe.encode(text, allowed_special)


    def decode(self, tokens: list[int], errors: str = "replace") -> str:
        return self._core_bpe.decode_bytes(tokens).decode("utf-8", errors=errors)


    @functools.cached_property
    def special_tokens_set(self) -> set[str]:
        return set(self._special_tokens.keys())



