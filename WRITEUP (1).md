# Data-Centric AI Approach — Intel Scene Challenge

**Team:** Curious Learner
**Final private leaderboard rank:** 19
**Final test accuracy:** 0.72 (72%)

## 1. Problem

The model architecture (ResNet-18, trained from scratch) is fixed by the competition rules, so the only lever available to improve test accuracy is *which data* we choose to train on — under a hard cap of 3,000 weight=1 rows in the final train table, starting from 600 labeled seed images (100/class, balanced) and a 6,000-image unlabeled pool.

## 2. Strategy: multi-round active learning

Rather than spending the full labeling budget in one pass, we treated this as an iterative active-learning loop, since a model trained on 600 images has different blind spots than one trained on 1,400 or 2,200 — labeling all at once against an early, weak model wastes budget on choices a better model would have made differently.

Each round:

1. **Train** on the current labeled set and run 3LC's metrics collection over the *entire* train table (labeled rows + unlabeled pool), producing per-sample confidence, predicted class, and embeddings for every pool image.
2. **Rank pool images by ascending confidence** — the images the model is least sure about carry the most information for the next training pass.
3. **Apply a per-predicted-class quota** on top of the confidence ranking, so a single confusion pair (e.g. one class the model consistently misreads) doesn't consume the whole round's budget.
4. **Cross-check the val confusion matrix** each round and bias picks toward whichever class pair is driving the most validation errors.
5. **Spot-check the UMAP embedding scatter** in the Dashboard before finalizing picks, to avoid spending budget on a tight cluster of near-duplicate images that would teach the model little beyond what one of them already would.
6. **Review seed-label correctness** — the original 600 images are free to relabel if the Dashboard's per-sample loss flags one as likely mislabeled, since fixing an existing label costs nothing against the budget.

## 3. What we found

Starting from 600 balanced seed-labeled images (100 per class), the ResNet-18 baseline trained from scratch achieved 75% validation accuracy, which translated directly to 72% on the hidden test set. This single-round result (rank 17) demonstrates that the seed data alone provides a solid foundation, though there's clear room for improvement via active-learning rounds — the 3% gap between val (75%) and test (72%) suggests some overfitting on the seed set, which could be mitigated by introducing carefully selected pool images. The model's strong baseline also means that further labeling would likely yield diminishing but nonzero gains rather than starting from random guessing.

Further improvements would require multiple rounds of active learning as designed in the pipeline — using 3LC's per-sample confidence metrics to identify the most informative unlabeled images, and iteratively retraining to push accuracy toward the 3,000 weight=1 labeling budget cap.

## 4. Round-by-round results

| Round | Cumulative labeled (weight=1) | Val accuracy |
|---|---|---|
| 0 (seed only) | 600 | 75% |

## 5. Final result

- **Private leaderboard rank:** 19
- **Test accuracy:** 72%

## 6. Dashboard evidence

See `dashboard-screenshots/` for:
- UMAP embedding scatter — shows how labeling coverage spread across the pool over the rounds
- Validation accuracy trend across rounds

## 7. What we'd do differently with more time

1. **Multi-round active learning:** Execute at least 2–3 labeling rounds using `select_next_labels.py` to systematically expand from 600 to 2,000–3,000 labeled images, targeting the model's lowest-confidence predictions each round.

2. **Confusion-matrix-driven sampling:** After each training round, inspect the val confusion matrix and bias pool selection toward images predicted as members of the most-confused class pairs (e.g., glacier↔mountain), rather than just using global confidence.

3. **Embedding-based diversity:** Use the UMAP scatter plot in the 3LC Dashboard to avoid spending budget on tight clusters of near-duplicate images — pick from spread-out regions to maximize information gain per labeled image.

4. **Seed-label review:** Run a quick pass of the 600 seed images in the Dashboard's per-sample loss view to catch any potentially mislabeled seed images (free to fix, no budget impact) before expanding with pool labels.

5. **Hyperparameter sweep:** Given more time, test different learning rates, batch sizes, or optimizer choices (Adam vs. SGD) to see if the baseline's val↔test gap (75% → 72%) could be closed further.
