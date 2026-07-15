# Texas Hold'em Agents ♠️

Reinforcement learning agents for **Texas Hold'em** — the full-scale poker environment of this project.

## 🎮 The Environment

Texas Hold'em is played with a standard **52-card deck**. Each player receives 2 private (hole) cards, and up to 5 community cards are dealt across four betting rounds (pre-flop, flop, turn, river). The enormous number of possible game states, combined with hidden opponent cards, makes this a challenging imperfect-information environment for reinforcement learning.

Compared to the Leduc experiments (see the [`Leduc/`](../Leduc) folder), this environment tests how well the same ideas scale to a realistic poker setting where exact solutions are intractable and function approximation / abstraction becomes necessary.

## 🧠 Agents Implemented

- [Agent 1 — e.g., DQN agent]
- [Agent 2 — e.g., NFSP agent]
- [Baseline — e.g., random / rule-based agent]

Agents are trained via self-play and evaluated by their average payoff against baseline and trained opponents.

## 📁 Contents

| File | Description |
|------|-------------|
| [notebook name].ipynb | Training and evaluation of the Texas Hold'em agents |
| [other files] | [description] |

## 🚀 How to Run

From the repository root:

```bash
pip install rlcard torch numpy matplotlib jupyter
cd Texas_Holdem
jupyter notebook
```

Open the notebook and run all cells (`Kernel → Restart & Run All`). The notebooks can also be run in [Google Colab](https://colab.research.google.com/).

## 📊 Results

Training curves, win rates, and payoff comparisons are generated inside the notebooks. See the **Final Project Report** at the repository root for the full analysis and discussion.
