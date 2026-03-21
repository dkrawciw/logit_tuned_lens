from transformer_lens import HookedTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

"""Plot setup"""
sns.set_style("whitegrid")
sns.set_color_codes(palette="colorblind")

plt.rcParams.update({
	"text.usetex": False,  # keep False to avoid requiring a LaTeX installation
	"mathtext.fontset": "cm",  # Computer Modern (LaTeX-like)
	"font.family": "serif",
	"font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.labelsize": 14,      # increase axis label size
    "axes.titlesize": 16,
    "xtick.labelsize": 14,     # increase tick / bin label size
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

# 1. Setup Device and Model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Use gpt2-small for faster training/consistency
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
model.eval() # We don't want to train the GPT-2 model itself

# 2. Define the Tuned Lens Module
class TunedLens(nn.Module):
    def __init__(self, n_layers, d_model):
        super().__init__()
        # Each layer gets its own linear transformation + bias
        self.lenses = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])
        # Initialize as Identity: weight = Eye, bias = Zero
        for lens in self.lenses:
            nn.init.eye_(lens.weight)
            nn.init.zeros_(lens.bias)

    def forward(self, resid, layer_idx):
        # 1. Apply the learned "correction"
        corrected = self.lenses[layer_idx](resid)
        # 2. Apply model's final LayerNorm and Unembed to get logits
        return model.unembed(model.ln_final(corrected))

# 3. Training Setup
lens = TunedLens(model.cfg.n_layers, model.cfg.d_model).to(device)
# Use a higher learning rate for faster movement
optimizer = torch.optim.Adam(lens.parameters(), lr=5e-4)

# 4. Prepare Dataset
print("Loading dataset...")
train_ds = load_dataset("NeelNanda/pile-10k", split="train[:1000]") # More data

# 5. Optimized Training Loop
def train():
    lens.train()
    batch_size = 12
    seq_len = 64
    epochs = 1 # More epochs
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        pbar = tqdm(range(0, len(train_ds), batch_size))
        for i in pbar:
            batch_texts = train_ds[i : i + batch_size]["text"]
            tokens = model.to_tokens(batch_texts, truncate=True)[:, :seq_len].to(device)
            input_tokens = tokens[:, :-1]
            
            with torch.no_grad():
                final_logits, cache = model.run_with_cache(input_tokens)
                # For KL Divergence, our target is the full probability distribution
                target_probs = F.softmax(final_logits, dim=-1)
            
            total_loss = 0
            for layer_idx in range(model.cfg.n_layers):
                resid = cache["resid_post", layer_idx]
                logits = lens(resid, layer_idx)
                
                # KLDivLoss expects log-probabilities for the input
                log_probs = F.log_softmax(logits, dim=-1)
                
                # reduction='batchmean' is mathematically correct for KL Div
                loss = F.kl_div(
                    log_probs.view(-1, log_probs.size(-1)), 
                    target_probs.view(-1, target_probs.size(-1)),
                    reduction='batchmean'
                )
                total_loss += loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch} | Loss: {total_loss.item():.4f}")

    # Check how much we moved from identity
    with torch.no_grad():
        diff = torch.norm(lens.lenses[5].weight - torch.eye(model.cfg.d_model).to(device))
        print(f"Weight movement from identity (Layer 5): {diff.item():.4f}")

    torch.save(lens.state_dict(), "tuned_lens_gpt2_small.pt")
    print("Training complete. Model saved.")


def plot_tuned_lens(prompt, lens_model, top_k=10, save_path="plots/tuned_lens.svg"):
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    lens_model.eval()
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)

    last_pos = -1
    n_layers = model.cfg.n_layers
    
    prob_matrix = np.zeros((top_k, n_layers))
    label_matrix = []

    for layer in range(n_layers):
        resid = cache["resid_post", layer]
        with torch.no_grad():
            layer_logits = lens_model(resid, layer)
            
        layer_probs = layer_logits.softmax(dim=-1)[0, last_pos]
        top_probs, top_indices = layer_probs.topk(top_k)
        
        layer_labels = []
        for i in range(top_k):
            token_str = model.to_string(top_indices[i]).strip()
            if not token_str: token_str = f"ID:{top_indices[i].item()}"
            prob_val = top_probs[i].item()
            prob_matrix[i, layer] = prob_val
            # layer_labels.append(f"{token_str}\n{prob_val:.2f}")
            layer_labels.append(f"{token_str}")
        label_matrix.append(layer_labels)

    annot_data = np.array(label_matrix).T

    plt.figure(figsize=(15, 8))
    sns.heatmap(
        prob_matrix, 
        annot=annot_data, 
        fmt="", 
        cmap="rocket",
        xticklabels=[f"L{i}" for i in range(n_layers)],
        yticklabels=[f"Rank {i+1}" for i in range(top_k)],
        cbar_kws={'label': 'Probability'}
    )
    
    plt.title(f"Tuned Lens Evolution: Top {top_k} tokens per layer\nPrompt: \"{prompt}\"", pad=20)
    plt.xlabel("Model Layers")
    plt.ylabel("Token Probability Rank")
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Tuned Lens plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # train() # Uncomment to re-train
    prompt = "The best state in the US is"
    

    # Load and visualize
    lens = TunedLens(model.cfg.n_layers, model.cfg.d_model).to(device)
    if os.path.exists("tuned_lens_gpt2_small.pt"):
        lens.load_state_dict(torch.load("tuned_lens_gpt2_small.pt", map_location=device))
        print("Loaded existing tuned lens weights.")
        
        plot_tuned_lens(prompt, lens)
    else:
        print("No weights found. Running training first...")
        train()
        plot_tuned_lens(prompt, lens)
