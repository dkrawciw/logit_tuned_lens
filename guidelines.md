# Project Guidelines: Logit and Tuned Lens Analysis

This document outlines the roadmap for completing your mechanistic interpretability project on GPT-2 using the Logit Lens and Tuned Lens techniques.

---

## 1. Conceptual Roadmap

### Logit Lens
- **Concept:** Take the internal state (residual stream) at layer $L$ and immediately apply the final LayerNorm and Unembedding (LM Head) to it.
- **Goal:** See what the model "thinks" the next token is halfway through its computation.
- **Analogy:** Checking a student's rough draft at various stages of writing.

### Tuned Lens
- **Concept:** Instead of just using the final LM Head, train a small "adapter" (a linear layer + bias) for each layer to map that layer's residual stream to the output vocabulary.
- **Goal:** Correct for "residual stream shift." The residual stream at early layers isn't "meant" to be read by the LM Head yet; the Tuned Lens learns to "translate" it.
- **Analogy:** Having a translator who understands the student's shorthand notes and can tell you what they are likely to write in the final draft.

---

## 2. Phase 1: Refining the Logit Lens (`src/logit.py`)

Your current code is a good start. To make it "submission-ready":
- **Efficiency:** Use `model.to_tokens` and `model.run_with_cache` effectively.
- **Visualization:** Instead of just printing top-k tokens, create a **Logit Lens Heatmap**.
    - X-axis: Token position in the prompt.
    - Y-axis: Layer number.
    - Color/Text: Probability of the "correct" token or the top predicted token at that layer.

---

## 3. Phase 2: Building the Tuned Lens (`src/tuned.py`)

Your current implementation needs a few corrections for efficiency and accuracy:
- **Objective:** The Tuned Lens should try to predict the **model's final output logits** (or the ground truth next token). 
- **Loss Function:** Use `KL Divergence` (between lens output and final model output) or `Cross Entropy` (with the next token). Avoid `MSE` on raw logits as it doesn't handle probability distributions well.
- **Training Efficiency:**
    - Don't iterate layers inside the dataset loop. Instead, pass the entire cache of residual streams to the `TunedLens` module and update all lenses in one forward/backward pass per batch.
    - Use a smaller model (like `gpt2-small`) for faster iteration during development.

---

## 4. Phase 3: Visualization & Analysis

For the "nice visualizations" requirement:
- **Comparison Heatmaps:** Show the Logit Lens vs. Tuned Lens side-by-side for the same prompt. You'll notice the Tuned Lens is much "smoother" and reaches the correct prediction earlier.
- **Entropy Plot:** Plot the entropy of the predicted distributions across layers. Usually, entropy decreases as the model becomes more certain.

---

## 5. Phase 4: Discussion & Reflection

In your final report/GitHub README, address these points:
- **What do we find?** (e.g., "The model decides on the next token's category early, but fine-tunes the specific word later.")
- **Why is this useful?** (e.g., "Debugging model behavior, understanding where a model 'goes wrong' in a chain of thought.")
- **What problems does it have?** (e.g., "Tuned lenses might 'hallucinate' predictions that aren't actually in the residual stream yet, just because it's trained to find them.")

---

## 6. Recommended Task List

1. [ ] **Fix `src/tuned.py`**: Update the loss function and training loop.
2. [ ] **Add Visualization**: Create a notebook or script that generates heatmaps using `matplotlib` or `plotly`.
3. [ ] **Run Experiments**: Test on interesting prompts (e.g., factual recall, logic, grammar).
4. [ ] **Write Discussion**: Document your findings in `README.md` or a new `DISCUSSION.md`.
