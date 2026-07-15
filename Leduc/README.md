# Leduc Hold'em Agents ♣️

Reinforcement learning agents for **Leduc Hold'em** — the simplified poker environment of this project.

## 🎮 The Environment

Leduc Hold'em is a small-scale research variant of poker, played with only **6 cards** (two suits, three ranks — e.g., J, Q, K of two suits). Each player receives 1 private card, a single community card is revealed, and there are two betting rounds.

Its tiny state space makes it ideal for studying imperfect-information games: algorithms can be trained quickly, results are easy to analyze, and learned strategies can even be compared against (near-)optimal play. It serves as the prototyping ground before scaling up to full [Texas Hold'em](../Texas_Holdem).

## 🧠 Agents Implemented

- [Agent 1 — e.g., CFR agent]
- [Agent 2 — e.g., Q-learning / DQN agent]
- [Baseline — e.g., random / rule-based agent]

Agents are trained via self-play and evaluated by their average payoff against baseline and trained opponents.

## 📁 Contents

| File | Description |
|------|-------------|
| [notebook name].ipynb | Training and evaluation of the Leduc Hold'em agents |
| [other files] | [description] |

## 🚀 How to Run

From the repository root:

```bash
pip install rlcard torch numpy matplotlib jupyter
cd Leduc
jupyter notebook
```

Open the notebook and run all cells (`Kernel → Restart & Run All`). The notebooks can also be run in [Google Colab](https://colab.research.google.com/).

## 📊 Results

Training curves, win rates, and payoff comparisons are generated inside the notebooks. See the **Final Project Report** at the repository root for the full analysis and discussion.
