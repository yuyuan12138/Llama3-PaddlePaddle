from model import Llama3
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
from tokenizer import tokenizer
import argparse

def read_json(path):
    english = []
    chinese = []
    print("start_read")
    with open(f'translation2019zh/translation2019zh_{path}.json', 'r', encoding='utf-8') as f:
        while True:
            try:
                line = eval(f.readline())
                english.append(line['english'])
                chinese.append(line['chinese'])
            except:
                break
    print("end_read")
    return english, chinese

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练参数")

    parser.add_argument('device', type=str, help="输入 gpu:0 或 cpu", default="gpu:0")
    parser.add_argument('if_train', type=bool, help="是否使用训练集(较大)", default=False)
    parser.add_argument('if_valid', type=bool, help='是否使用验证集(较小)', default=True)

    args = parser.parse_args()
    
    paddle.device.set_device(args.device)
    
    if args.if_train == True and args.if_valid == False:
        english_train, chinese_train = read_json("train")
    if args.if_train == False and args.if_valid == True:
        english_train, chinese_test = read_json("valid")
    if args.if_train and args.if_valid:
        english_train, chinese_train = read_json("valid")
        
    model = Llama3()
    train_size = len(english_train)

    loss_fn = nn.CrossEntropyLoss()

    optim = paddle.optimizer.AdamW(learning_rate=0.0001, parameters=model.parameters())

    for idx, text in enumerate(english_train):
        optim.clear_grad()
        print(round(idx / train_size, 3))
        tokens = [128000] + tokenizer.encode(text)
        tokens_ = tokens[:-2]
        re = tokens[-2:-1]
        next_index, output = model(tokens_)
        output = paddle.unsqueeze(output, axis=0).to(paddle.float32)
        loss = loss_fn(output, paddle.to_tensor(re))
        loss.backward()
        print(tokenizer.decode(paddle.unsqueeze(next_index, axis=0)), tokenizer.decode(re))
        print(f"loss: {loss.item()}")


    
    
