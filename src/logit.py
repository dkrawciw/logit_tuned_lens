from transformer_lens import HookedTransformer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

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

# Device setup
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small", device=device)

def plot_logit_evolution(prompt: str, top_k: int = 10, save_path: str = "plots/logit_lens.svg"):
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Run model with cache
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)

    # We only care about the NEXT token prediction (last position)
    last_pos = -1
    n_layers = model.cfg.n_layers
    
    # Data structures to hold results
    # We will track the TOP K tokens AT EACH LAYER
    prob_matrix = np.zeros((top_k, n_layers))
    label_matrix = [] # This will be our annotations (Token + Prob)

    for layer in range(n_layers):
        resid = cache["resid_post", layer]
        layer_logits = model.unembed(model.ln_final(resid))
        layer_probs = layer_logits.softmax(dim=-1)[0, last_pos]
        
        # Get top K for THIS specific layer
        top_probs, top_indices = layer_probs.topk(top_k)
        
        layer_labels = []
        for i in range(top_k):
            token_str = model.to_string(top_indices[i]).strip()
            # Clean up empty/special tokens for display
            if not token_str: token_str = f"ID:{top_indices[i].item()}"
            
            prob_val = top_probs[i].item()
            prob_matrix[i, layer] = prob_val
            # layer_labels.append(f"{token_str}\n{prob_val:.2f}")
            layer_labels.append(f"{token_str}")
            
        label_matrix.append(layer_labels)

    # Transpose label_matrix to match (top_k, n_layers) shape for heatmap
    # Current label_matrix is [layer][rank], we need [rank][layer]
    annot_data = np.array(label_matrix).T

    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Use seaborn for the heatmap
    sns.heatmap(
        prob_matrix, 
        annot=annot_data, 
        fmt="", 
        cmap="viridis",
        xticklabels=[f"L{i}" for i in range(n_layers)],
        yticklabels=[f"Rank {i+1}" for i in range(top_k)],
        cbar_kws={'label': 'Probability'}
    )
    
    plt.title(f"Logit Evolution: Top {top_k} tokens per layer\nPrompt: \"{prompt}\"", pad=20)
    plt.xlabel("Model Layers")
    plt.ylabel("Token Probability Rank")
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    prompt = "The best state in the US is"
    plot_logit_evolution(prompt)
