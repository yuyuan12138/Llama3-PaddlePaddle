from model import Llama3
import paddle
import paddle.nn as nn
from tokenizer import tokenizer
from utils import read_json, reshape_chinese_english_token

if __name__ == "__main__":

    # 使用CPU训练
    # paddle.device.set_device('cpu')

    # 使用GPU训练
    paddle.device.set_device('gpu:0')

    # 读入数据集
    english_train, chinese_train = read_json("train")   # 训练集
    
    english_valid, chinese_valid = read_json('valid')   # 验证集
        
    model = Llama3()
    # train_size = len(english_train)

    loss_fn = nn.MSELoss()

    optim = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
    
    model.train()
    for idx, (english, chinese) in enumerate(zip(english_train, chinese_train)):
        optim.clear_grad()
        tokens_english = [128000] + tokenizer.encode(english)
        tokens_chinese = [128000] + tokenizer.encode(chinese)

        tokens_english, tokens_chinese = reshape_chinese_english_token(paddle.to_tensor(tokens_english, dtype="float32"), 
                                                                       paddle.to_tensor(tokens_chinese, dtype='float32'))
        
        pre_index, ouput = model(tokens_english)
        loss = loss_fn(ouput, tokens_chinese)
        loss.backward()
        print(f"loss: {loss.item()}")
        if idx % 10000 == 0 and idx != 0:
            paddle.save(model, f"model_{idx}")
        print(tokenizer.decode(pre_index), chinese)
    
    with paddle.no_grad():
        for idx, (english, chinese) in enumerate(zip(english_valid, chinese_valid)):
            optim.clear_grad()
            tokens_english = [128000] + tokenizer.encode(english)
            tokens_chinese = [128000] + tokenizer.encode(chinese)

            tokens_english, tokens_chinese = reshape_chinese_english_token(paddle.to_tensor(tokens_english, dtype="float32"), 
                                                                           paddle.to_tensor(tokens_chinese, dtype='float32'))

            pre_index, ouput = model(tokens_english)
            print(tokenizer.decode(pre_index), chinese)
    


    
    
