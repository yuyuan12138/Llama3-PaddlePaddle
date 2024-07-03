import bpe
import json
import tqdm

def read_tokenizer_model():
    vocab = {}
    with open("./state_dict/tokenizer.model", "r") as f:
        while True:
            try:
                tmp = f.readline()
                split_str = tmp.split()
                vocab[split_str[0]] = int(split_str[1])
            except:
                break
    return vocab

def test(texts, vocab=None):
    special_tokens = {
            "<|begin_of_text|>": 265
            }
    textes = "FloydHub is the fastest way to build, train and deploy deep learning models. Build deep learning models in the cloud. Train deep learning models." 

    tokenizer = bpe.RegexTokenizer()

    tokenizer.register_special_tokens(special_tokens=special_tokens)
    texts_size = len(texts)
    tokenizer.vocab = vocab
    tokenizer.train(text=textes, vocab_size=len(vocab))
    # for idx, text in enumerate(texts):
    #     tokenizer.train(text=text, vocab_size=1024)
    #     print(f"{round(idx / texts_size * 100, 2)}%")
    print(tokenizer.encode(textes))
    print(tokenizer.decode(tokenizer.encode(textes)))
    # print(tokenizer.vocab)

def read_json():
    english = []
    chinese = []
    with open('translation2019zh/translation2019zh_valid.json', 'r', encoding='utf-8') as f:
        while True:
            try:
                line = eval(f.readline())
                english.append(line['english'])
                chinese.append(line['chinese'])
            except:
                break
    return english, chinese

if __name__ == "__main__":
    # test()
    english, chinese = read_json()
    test(english, read_tokenizer_model())
    # print(read_tokenizer_model())

    
    
