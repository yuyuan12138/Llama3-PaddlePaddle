import paddle
import paddle.nn as nn
from model import Llama3
from tokenizer import tokenizer
import paddle.optimizer as optim

class Train:
    def __init__(self, prompt, lr=1e-4, ) -> None:
        self.model = Llama3()
        self.tokenizer = tokenizer
        self.tokens = [128000] + tokenizer.encode(prompt)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(lr=lr, parameters=self.model.parameters())
    
    def train(self, epochs):
        assert type(epochs) is int, "Set epochs in type of int"

        self.model.train()
        for epoch in range(epochs):
            print(f" --- epoch {epoch + 1} start")
            
            self.optimizer.clear_gradients()

            next_token = self.model(self.tokens)

            loss = self.loss_fn(next_token, 42)        
            loss.backward()

            self.optimizer.step() 


if __name__ == "__main__":
    prompt = "the answer to the ultimate question of life, the universe, and everything is "
    train = Train(prompt=prompt)
    
    train.train(1)
