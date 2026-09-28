from tinygrad import Tensor, nn

class CausalSelfAttention:
    def __init__(self, dim, heads, dropout=0.0):
        assert dim % heads == 0, "dim debe ser divisible por heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        b, t, c = x.shape

        q = self.q(x).reshape(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(b, t, self.heads, self.head_dim).permute(0, 2, 1, 3)

        y = q.scaled_dot_product_attention(
            k, v,
            dropout_p=self.dropout,
            is_causal=True
        )

        y = y.permute(0, 2, 1, 3).reshape(b, t, c)
        return self.proj(y)

class FeedForward:
    def __init__(self, dim, ff_dim, dropout=0.0):
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
        self.dropout = dropout

    def __call__(self, x):
        x = self.fc1(x).gelu()
        x = self.fc2(x)
        return x.dropout(self.dropout)

class TransformerBlock:
    def __init__(self, dim, heads, ff_dim, dropout=0.0):
        self.norm1 = nn.RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.norm2 = nn.RMSNorm(dim)
        self.ff = FeedForward(dim, ff_dim, dropout)
        self.dropout = dropout

    def __call__(self, x):
        x = x + self.attn(self.norm1(x)).dropout(self.dropout)
        x = x + self.ff(self.norm2(x))
        return x

class VikoGPT:
    def __init__(self, cfg):
        m = cfg["model"]
        self.vocab_size = int(m["vocab_size"])
        self.context = int(m["context"])
        self.dim = int(m["dim"])
        self.heads = int(m["heads"])
        self.layers_n = int(m["layers"])
        self.ff_dim = int(m["ff_dim"])
        self.dropout = float(m.get("dropout", 0.0))

        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)
        self.position_embedding = nn.Embedding(self.context, self.dim)
        self.blocks = [
            TransformerBlock(self.dim, self.heads, self.ff_dim, self.dropout)
            for _ in range(self.layers_n)
        ]
        self.norm = nn.RMSNorm(self.dim)
        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)

    def __call__(self, idx):
        b, t = idx.shape
        if t > self.context:
            raise ValueError(f"Secuencia {t} > contexto {self.context}")

        pos = Tensor.arange(t).reshape(1, t).expand(b, t)
        x = self.token_embedding(idx) + self.position_embedding(pos)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.lm_head(x)
