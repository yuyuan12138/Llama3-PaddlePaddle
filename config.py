import json
import paddle

class Config:
    def __init__(self) -> None:
        """
        从 JSON 文件加载配置参数并初始化 Config 类的实例。
        """
        # 打开并读取 JSON 配置文件
        with open("state_dict/original_params.json", "r") as f:
            config = json.load(f)

        # 将读取的配置信息设置为类的属性
        self.dim = config["dim"]  # 模型的维度
        self.n_layers = config["n_layers"]  # 模型中层数
        self.n_heads = config["n_heads"]  # 注意力机制的头数
        self.n_kv_heads = config["n_kv_heads"]  # 键值对头数
        self.vocab_size = config["vocab_size"]  # 词汇表大小
        self.multiple_of = config["multiple_of"]  # 用于确保某些参数是此值的倍数
        self.ffn_dim_multiplier = config["ffn_dim_multiplier"]  # 前馈网络维度的乘数
        self.norm_eps = config["norm_eps"]  # 归一化层的 epsilon 值，防止除以零
        self.rope_theta = paddle.to_tensor(config["rope_theta"])  # RoPE 编码的 theta 值

# 创建 Config 实例
config = Config()
