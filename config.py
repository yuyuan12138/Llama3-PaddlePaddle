import json
import paddle

class Config:
    def __init__(self) -> None:
        with open("state_dict/original_params.json", "r") as f:
            config = json.load(f)

        self.dim = config["dim"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.n_kv_heads = config["n_kv_heads"]
        self.vocab_size = config["vocab_size"]
        self.multiple_of = config["multiple_of"]
        self.ffn_dim_multiplier = config["ffn_dim_multiplier"]
        self.norm_eps = config["norm_eps"]
        self.rope_theta = paddle.to_tensor(config["rope_theta"])

config = Config()