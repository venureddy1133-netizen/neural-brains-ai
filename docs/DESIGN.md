# Design notes — v2.0

- `TextEncoder`: deterministic hashed unigram and bigram features.
- `NeuralNetwork`: tanh hidden layer, softmax output, cross-entropy gradient.
- `BrainAgent`: bootstraps only when no model file exists; otherwise loads saved parameters.
- `MemoryStore`: SQLite log of predictions and labelled teaching examples.
- `dashboard`: local static HTML snapshot.

## Persistence
`brain_model.json` stores trainable parameters and update count. It is written to a temporary file then renamed. SQLite stores experiences separately. Weights are saved directly; they are not reconstructed from memory rows.

## Output
The three-class softmax scores sum to 1. They are relative model scores, not calibrated probabilities or validated confidence.

## Privacy
Data remains in the chosen local data directory. JSON and SQLite are plaintext, not encrypted.
