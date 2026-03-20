from transformer_lens import HookedTransformer
import torch
import torch.nn as nn

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

model = HookedTransformer.from_pretrained("gpt2", dtype=torch.float32)
model.eval().to(device)

class TunedLens(torch.nn.Module):
    """One affine probe per layer, each mapping residual stream -> residual stream."""
    def __init__(self, n_layers: int, d_model: int):
        super().__init__()
        self.lenses = torch.nn.ModuleList([
            torch.nn.Linear(d_model, d_model, bias=True)
            for _ in range(n_layers)
        ])
        for lens in self.lenses:
            torch.nn.init.eye_(lens.weight)
            torch.nn.init.zeros_(lens.bias)

    def forward(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply lens[layer_idx], then model's LN + unembed -> logits."""
        corrected = self.lenses[layer_idx](hidden)
        return model.unembed(model.ln_final(corrected))

lens = TunedLens(d_model=model.cfg.d_model, n_layers=model.cfg.n_layers)
optimizer = torch.optim.Adam(lens.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Get a dataset to train the tuned lens on. The Pile is a common choice for language modeling tasks.
from datasets import load_dataset
ds = load_dataset("NeelNanda/pile-10k", split="train[:1000]")

def tokenize(example):
    return {"tokens": model.to_tokens(example["text"])[0]}

tokenized_ds = ds.map(tokenize)

# print(tokenized_ds[0]["text"])

for i in range(model.cfg.n_layers):
    for example in tokenized_ds:
        tokens = torch.tensor(example["tokens"]).to(device)
        with torch.no_grad():
            outputs = model(tokens.unsqueeze(0))
            hidden = outputs.hidden_states[i].squeeze(0)  # (seq_len, d_model)
            target = model.unembed(model.ln_final(hidden))  # (seq_len, vocab_size)
        X = hidden
        y = target
        preds = lens(X, i)
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(lens.lenses[0].weight)