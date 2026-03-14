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

import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout

        # On combine les projections Q, K, V en une seule couche pour plus de vitesse
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape

        # 1. Projection QKV groupée (plus efficace que 3 linéaires séparés)
        qkv = self.qkv_proj(x) # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # 2. Utilisation du noyau natif "Flash Attention / Memory Efficient"
        # Cette ligne remplace TOUTE votre logique matmul + scale + softmax + dropout
        attn = (Q @ K.transpose(-2,-1)) / math.sqrt(self.d_head)
        attn = attn.softmax(-1)
        out = attn @ V

        # 3. Reconstruction et sortie
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
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
        self.register_buffer("positions", torch.arange(max_seq_len).unsqueeze(0))

    def _init_weights(self, beta: float):
        """
        Initialisation spécifique DeepNorm adaptée au MHA optimisé.
        """
        nn.init.normal_(self.token_emb.weight, mean=0, std=1.0)
        nn.init.normal_(self.pos_emb.weight,   mean=0, std=1.0)

        for layer in self.layers:
            # Accès au MHA optimisé (qkv_proj et out_proj)
            nn.init.normal_(layer.attn.qkv_proj.weight, mean=0, std=beta)
            nn.init.normal_(layer.attn.out_proj.weight, mean=0, std=beta)
            
            # Accès au FeedForward (vérifiez les indices selon votre classe FF)
            # Généralement net[0] est le Linear d'entrée et net[3] le Linear de sortie
            nn.init.normal_(layer.ff.net[0].weight, mean=0, std=beta)
            nn.init.normal_(layer.ff.net[3].weight, mean=0, std=beta)

    def forward(
        self,
        tokens: torch.Tensor,          # (B, T)
        mask: torch.Tensor = None,     # (B, 1, 1, T) ou (B, 1, T, T)
    ) -> torch.Tensor:
        B, T = tokens.shape
        
        x = self.drop_emb(self.token_emb(tokens) + self.pos_emb(self.positions[:, :T]))
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
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.encoder(tokens, mask)   # (B, T, d_model)
        cls = x[:, 0, :]              # token [CLS] en position 0
        policy = self.head(cls)
        V = torch.tanh(self.value_head(cls))

        return V, policy         # (B, num_classes)


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

    def forward(self, policy, policy_val, V, eval_val):
        ce_loss  = self.ce(policy_val, policy)
        mse_loss = self.mse(V, eval_val)

        self.last_ce  = ce_loss.item()
        self.last_mse = mse_loss.item()
        print(ce_loss, mse_loss)

        return self.alpha * ce_loss + self.beta * mse_loss
        
class Eval_(nn.Module):
    def __init__(self):
        super().__init__()
        self.SoftMax = nn.Softmax(dim=-1)

    def forward(self, policy_val, mask):
        policy_val=self.SoftMax(torch.add(policy_val,mask))

        return policy_val
    

if __name__ == "__main__":
    torch.manual_seed(42)

    # Config legere
    cfg = dict(
        vocab_size  = 15,
        max_seq_len = 373,
        d_model     = 64,
        n_heads     = 2,
        n_layers    = 4,
        d_ff        = 128,
        dropout     = 0.1,
    )

    # Encoder seul
    encoder = DeepNormEncoder(**cfg)
    print(f"Encoder  - parametres : {count_params(encoder):,}")

    tokens = torch.randint(0, cfg["vocab_size"], (1, 373))           # batch=1, seq=1620
    tokens = torch.stack([tokens[0], tokens[0]])                      # batch=2, seq=1620
    out    = encoder(tokens)
    print(f"Encoder output shape : {out.shape}")            # (2, 4501)

    # Classifier
    mask = torch.full((1, 4500), -1e9)
    mask[0][[0,2]] = 0.0
    print(mask)
    clf = DeepNormClassifier(num_classes=4500, **cfg)
    print(f"Classifier - parametres : {count_params(clf):,}")

    # Inférence / évaluation
    clf.eval()
    with torch.no_grad():
        eval, policy = clf(tokens)

    print(f"Eval shape : {eval.shape}")                     # (2, 1)
    print(f"Policy shape : {policy.shape}")                 # (2, 4501)
    print(policy, eval)     

    torch.save(clf.state_dict(), os.getcwd() + "//PyXiangQi//models//weights//v0.pth")

    # Verification du gradient
    clf.train()
    eval, policy = clf(tokens)
    loss=CombinedLoss()
    tot=loss(policy,policy,eval,eval)
    tot.backward()

    print("Backprop OK")
    #print(eval)
    #print(nn.Softmax(policy))