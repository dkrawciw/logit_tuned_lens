# Logit/Tuned Lens Project Context

This project implements and explores the **Logit Lens** and **Tuned Lens** techniques for mechanistic interpretability of Transformer models, specifically targeting GPT-2.

## Project Overview

The goal of this project is to provide tools and visualizations for "peeking" into the internal layers of a Transformer model to see what it "thinks" at each stage of processing.

-   **Logit Lens:** Directly applies the model's final unembedding (LM Head) to the residual stream at each layer.
-   **Tuned Lens:** Enhances the Logit Lens by training a per-layer affine transformation (linear layer + bias) to more accurately map the intermediate residual stream to the output vocabulary space.

## Architecture & Key Files

-   `src/logit.py`: Implementation of the basic Logit Lens using the `transformer_lens` library. It demonstrates how the model's predictions evolve layer by layer.
-   `src/tuned.py`: Implementation of the Tuned Lens. Includes a `TunedLens` PyTorch module and a training script that uses the `pile-10k` dataset to optimize the lenses.
-   `notebooks/in_class_exp.ipynb`: Experimental notebook containing visualizations and exploratory code for layer-wise analysis.
-   `main.py`: Entry point for the project (currently a placeholder).

## Technical Stack

-   **Language:** Python 3.12+
-   **Deep Learning:** [PyTorch](https://pytorch.org/)
-   **Interpretability:** [transformer_lens](https://github.com/TransformerLensOrg/TransformerLens)
-   **Data:** [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

## Building and Running

### Prerequisites

Ensure you have Python 3.12+ installed. The project uses `pyproject.toml` for dependency management.

### Installation

```bash
# TODO: Install dependencies (requires-python >= 3.12)
# Example: pip install torch transformer_lens datasets
```

### Running Experiments

-   **Logit Lens:**
    ```bash
    python src/logit.py
    ```
-   **Tuned Lens Training:**
    ```bash
    python src/tuned.py
    ```

## Development Conventions

-   **Source Code:** All core logic resides in the `src/` directory.
-   **Notebooks:** Use the `notebooks/` directory for interactive experiments and visualizations.
-   **Model Loading:** Prefers `HookedTransformer` from `transformer_lens` for ease of access to internal activations (the "cache").
-   **Device Support:** `src/tuned.py` automatically detects and uses MPS (Apple Silicon), CUDA, or CPU.
