from pathlib import Path
# import tiktoken_ as tiktoken
import tiktoken
import base64
import paddle
# from regexs import RegexTokenizer
import bpe

def load_tiktoken_bpe(cache_path):
    with open(cache_path, "rb") as f:
            contents = f.read()
    return {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in contents.splitlines() if line)
        }

class Tokenizer:
    def __init__(self) -> None:
        tokenizer_path = "state_dict/tokenizer.model"
        GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        LLAMA3_SPLIT_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
        mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
        # print(mergeable_ranks)
        self.tokenizer = tiktoken.Encoding(
            name = "sss",
            pat_str=LLAMA3_SPLIT_PATTERN,
            # pat_str=GPT4_SPLIT_PATTERN,
            # pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
        )
tokenizer = Tokenizer().tokenizer
    
if __name__ == "__main__":
    # tokenizer = Tokenizer().tokenizer
    # prompt = "the answer to the ultimate question of life, the universe, and everything is "
    # tokens = [128000] + tokenizer.encode(prompt)
    # print(tokens)
    
    # tokens = paddle.to_tensor(tokens)
    # prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
    # print(prompt_split_as_tokens)
    llama_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()
    special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    LLAMA3_SPLIT_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # tokenizer = RegexTokenizer(GPT4_SPLIT_PATTERN)
    # tokenizer = Tokenizer().tokenizer
    tokenizer = bpe.RegexTokenizer(LLAMA3_SPLIT_PATTERN)
    tokenizer.register_special_tokens(special_tokens=special_tokens)
    prompt = llama_text
    # print(prompt)
    # print(tokenizer.encode(prompt))
    tokens = paddle.to_tensor(tokenizer.encode(prompt, set(special_tokens)))
    prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
    print(prompt_split_as_tokens)
    
