"""
DeepNorm Transformer (PyTorch) — version allégée
Référence : "DeepNet: Scaling Transformers to 1,000 Layers" (Wang et al., 2022)
https://arxiv.org/abs/2203.00555

Idée clé de DeepNorm :
  x = LayerNorm(alpha * x + SubLayer(x))
  avec une initialisation des poids mise à l'échelle par beta.

Pour un encoder de N couches :
  alpha = (2N)^(1/4)
  beta  = (8N)^(-1/4)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 1. Blocs de base
# ──────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q, K, V = split_heads(self.q(x)), split_heads(self.k(x)), split_heads(self.v(x))

        scale = math.sqrt(self.d_head)
        attn = (Q @ K.transpose(-2, -1)) / scale          # (B, H, T, T)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = self.dropout(F.softmax(attn, dim=-1))
        #attn = F.softmax(attn, dim=-1)
        
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
# 2. Couche DeepNorm  (cœur de l'architecture)
# ──────────────────────────────────────────────

class DeepNormEncoderLayer(nn.Module):
    """
    Variante DeepNorm d'une couche Transformer :
        x <- LayerNorm(alpha * x + SubLayer(x, ...))
    au lieu du Pre-LN ou Post-LN standard.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 alpha: float, dropout: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff   = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Bloc attention avec DeepNorm
        x = self.norm1(self.alpha * x + self.attn(x, mask))
        # Bloc feed-forward avec DeepNorm
        x = self.norm2(self.alpha * x + self.ff(x))
        return x


class DeepNormEncoder(nn.Module):
    """
    Encoder-only Transformer avec normalisation DeepNorm.

    Parametres recommandes pour rester leger :
        d_model = 128 ou 256
        n_heads = 4 ou 8
        n_layers = 6 a 12
        d_ff = 4 * d_model  (standard)
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int  = 128,
        n_heads: int  = 4,
        n_layers: int = 6,
        d_ff: int     = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers

        # Coefficients DeepNorm (formule encoder-only, Eq. 8 du papier)
        #   alpha = (2N)^(1/4)
        #   beta  = (8N)^(-1/4)
        alpha = (2 * n_layers) ** 0.25
        beta  = (8 * n_layers) ** (-0.25)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop_emb  = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DeepNormEncoderLayer(d_model, n_heads, d_ff, alpha, dropout)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)

        # Initialisation DeepNorm : mise a l'echelle par beta
        self._init_weights(beta)

    def _init_weights(self, beta: float):
        """
        Initialisation specifique DeepNorm :
        - Embeddings : normal(0, 1)
        - Poids des sous-couches (attention + FFN) : normal(0, beta)
        """
        nn.init.normal_(self.token_emb.weight, mean=0, std=1.0)
        nn.init.normal_(self.pos_emb.weight,   mean=0, std=1.0)

        for layer in self.layers:
            for module in [layer.attn.q, layer.attn.k, layer.attn.v,
                           layer.attn.out,
                           layer.ff.net[0], layer.ff.net[3]]:
                nn.init.normal_(module.weight, mean=0, std=beta)

    def forward(
        self,
        tokens: torch.Tensor,          # (B, T)
        mask: torch.Tensor = None,     # (B, 1, 1, T) ou (B, 1, T, T)
    ) -> torch.Tensor:
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)  # (1, T)

        x = self.drop_emb(self.token_emb(tokens) + self.pos_emb(positions))
        #x = self.token_emb(tokens) + self.pos_emb(positions)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm_out(x)   # (B, T, d_model)


# ──────────────────────────────────────────────
# 4. Tete d'évaluation (optionnelle)
# ──────────────────────────────────────────────

class DeepNormClassifier(nn.Module):
    """Wrapper : encoder DeepNorm + tete de classification sur le token [CLS]."""

    def __init__(self, num_classes: int, **encoder_kwargs):
        super().__init__()
        self.encoder = DeepNormEncoder(**encoder_kwargs)
        d_model = encoder_kwargs["d_model"]
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.encoder(tokens, mask)   # (B, T, d_model)
        cls = x[:, 0, :]              # token [CLS] en position 0
        head = self.head(cls)
        eval = torch.tanh(head[:,-1])
        policy = head[:,:-1]

        return eval, policy         # (B, num_classes)


# ──────────────────────────────────────────────
# 5. Demo / verification rapide
# ──────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        alpha : weight applied to Cross Entropy
        beta  : weight applied to MSE
        """
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.ce    = nn.CrossEntropyLoss()
        self.mse   = nn.MSELoss()

    def forward(self, policy, policy_val, eval, eval_val):
        ce_loss  = self.ce(policy_val, policy)
        mse_loss = self.mse(eval, eval_val)

        self.last_ce  = ce_loss.item()
        self.last_mse = mse_loss.item()
        print(ce_loss, mse_loss)

        return self.alpha * ce_loss + self.beta * mse_loss
    
class Eval_(nn.Module):
    def __init__(self):
        """
        alpha : weight applied to Cross Entropy
        beta  : weight applied to MSE
        """
        super().__init__()
        self.SoftMax = nn.Softmax(dim=-1)

    def forward(self, policy_val, mask):
        policy_val=self.SoftMax(torch.add(policy_val,mask))

        return policy_val
    

if __name__ == "__main__":
    torch.manual_seed(42)

    # Config legere
    cfg = dict(
        vocab_size  = 100,
        max_seq_len = 1620,
        d_model     = 128,
        n_heads     = 4,
        n_layers    = 6,
        d_ff        = 512,
        dropout     = 0.1,
    )

    # Encoder seul
    encoder = DeepNormEncoder(**cfg)
    print(f"Encoder  - parametres : {count_params(encoder):,}")

    tokens = torch.randint(0, cfg["vocab_size"], (1, 1620))           # batch=1, seq=1620
    tokens = torch.stack([tokens[0], tokens[0]])                      # batch=2, seq=1620
    out    = encoder(tokens)
    print(f"Encoder output shape : {out.shape}")            # (2, 4501)

    # Classifier
    mask = torch.full((1, 4500), float('-inf'))
    mask[0][[0,2]] = 0.0
    print(mask)
    clf = DeepNormClassifier(num_classes=4501, **cfg)
    print(f"Classifier - parametres : {count_params(clf):,}")

    # Inférence / évaluation
    clf.eval()
    with torch.no_grad():
        eval, policy = clf(tokens)

    print(f"Eval shape : {eval.shape}")                     # (2, 1)
    print(f"Policy shape : {policy.shape}")                 # (2, 4501)
    print(policy, eval)               

    torch.save(clf.state_dict(), os.getcwd() + "\\PyXiangQi\\models\\weights\\v0.pth")

    # Verification du gradient
    clf.train()
    eval, policy = clf(tokens)
    loss=CombinedLoss()
    tot=loss(policy,policy,eval,eval,mask=mask)
    tot.backward()

    print("Backprop OK")
    #print(eval)
    #print(nn.Softmax(policy))