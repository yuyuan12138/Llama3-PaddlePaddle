import regex

if __name__ == "__main__":
    text = "the answer to the ultimate question of life, the universe, and everything is "
    pattern = regex.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    tokens = pattern.findall(text)
    print(tokens)