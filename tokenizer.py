from pathlib import Path
# import tiktoken_ as tiktoken
import tiktoken
import base64
import paddle

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
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
        )
tokenizer = Tokenizer().tokenizer
    
if __name__ == "__main__":
    tokenizer = Tokenizer().tokenizer
    prompt = "the answer to the ultimate question of life, the universe, and everything is "
    tokens = [128000] + tokenizer.encode(prompt)
    print(tokens)
    
    tokens = paddle.to_tensor(tokens)
    prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
    print(prompt_split_as_tokens)
    
