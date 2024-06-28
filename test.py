import regex
import tiktoken
import json

class CoreBPE:
    def __init__(self, mergeable_ranks, special_tokens, pat_str):
        """
        初始化 BPE 编码器。
        
        :param mergeable_ranks: dict[bytes, int], 指定可以合并的字节对及其优先级。
        :param special_tokens: dict[str, int], 指定不应被合并的特殊标记的优先级。
        :param pat_str: str, 用于分割文本的正则表达式。
        """
        self.mergeable_ranks = mergeable_ranks
        self.special_tokens = special_tokens
        # print(special_tokens)
        self.pattern = regex.compile(pat_str)
    
    def encode(self, text, allowed_special=None):
        """
        对文本进行 BPE 编码。
        
        :param text: str, 待编码的原始文本。
        :return: list, 编码后的标记列表。
        """
        # text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
        # tokens = self.pattern.split(text)
        tokens = self.pattern.findall(text)
        encoded_tokens = []
        with open('tokenizer.json', 'r') as f:
            data = json.load(f)

        for token in tokens:
            if token in self.special_tokens:
                for i in data['add_tokens']:
                    if bytes(i["content"], encoding="utf-16") == bytes(token, encoding="utf-16"):
                        print(i)
                encoded_tokens.append(token)  # 直接将特殊标记添加到结果中
                continue
            
            encoded_tokens.append(token)
            for j in data['model']['vocab']:
                # print(j)
                if bytes(j, encoding="utf-16") == bytes(token, encoding="utf-16"):
                    print(j)
        
        return encoded_tokens
    
    def decode(self, tokens, allowed_special=None):
        """
        解码 BPE 编码的文本。
        
        :param tokens: list, BPE 编码后的标记列表。
        :return: str, 解码后的文本。
        """
        return ''.join(tokens)
    
    def add_new_pair(self, pair, rank):
        """
        添加新的合并字节对。
        
        :param pair: bytes, 新的字节对。
        :param rank: int, 字节对的合并优先级。
        """
        self.mergeable_ranks[pair] = rank

if __name__ == "__main__":
    # 初始化 BPE 编码器
    bpe_encoder = CoreBPE(
        mergeable_ranks={b'lo': 1, b'ol': 2}, 
        special_tokens={'<s>': 1, '</s>': 2}, 
        pat_str=r"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)",
    )

    # 编码文本
    encoded_text = bpe_encoder.encode("hello world <s>")
    print("Encoded:", encoded_text)

    # 解码文本
    decoded_text = bpe_encoder.decode(encoded_text)
    print("Decoded:", decoded_text)
