# Reinforcement Learning — Poker Agents ♠️♥️♣️♦️

Final project for the **Reinforcement Learning and Dynamic Optimization** course: training and evaluating poker-playing agents in two imperfect-information card game environments — **Leduc Hold'em** and **Texas Hold'em**.

Poker is a classic benchmark for reinforcement learning because, unlike games such as chess or Go, players have **incomplete information** about the state of the game (opponents' hidden cards). This makes it an excellent testbed for studying how RL agents learn strategies under uncertainty.

## 📁 Repository Structure

```
.
├── Leduc/                        # Agents & experiments for Leduc Hold'em (simplified poker)
├── Texas_Holdem/                 # Agents & experiments for Texas Hold'em
├── Report-Phase1.pdf             # Phase 1 project report
├── Final Project - Report.pdf    # Final project report
└── Final Presentation.pptx       # Final presentation slides
```

Each game folder contains its own README with details about the environment, the agents implemented, and how to run the experiments.

## 🎮 The Two Environments

| | **Leduc Hold'em** | **Texas Hold'em** |
|---|---|---|
| Deck size | 6 cards (2 suits × 3 ranks) | 52 cards |
| Complexity | Small — ideal for studying algorithms | Large state space — a real challenge |
| Purpose | Prototype and analyze agent behavior | Scale up to a realistic poker setting |

## 🧠 Approach

Agents are trained through self-play and evaluated against baseline opponents (e.g., random agents) as well as against each other. The implemented methods include:

- [Algorithm 1 — e.g., Q-learning / DQN]
- [Algorithm 2 — e.g., CFR (Counterfactual Regret Minimization)]
- [Algorithm 3 — e.g., NFSP / rule-based baselines]

Full details of the methodology, experiments, and results can be found in the **Final Project Report** and the **presentation slides**.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

```bash
pip install rlcard torch numpy matplotlib jupyter
```

> Adjust the list if the notebooks use different libraries.

### Running

1. Clone the repository:
   ```bash
   git clone https://github.com/Pavlos43/Reinforcment_Learning-Poker_Agents.git
   cd Reinforcment_Learning-Poker_Agents
   ```
2. Enter one of the environment folders (`Leduc/` or `Texas_Holdem/`) and open the notebooks with Jupyter, or run them in [Google Colab](https://colab.research.google.com/).

## 📄 Reports & Presentation

- **`Report-Phase1.pdf`** — initial phase: problem setup, environment, and first agents
- **`Final Project - Report.pdf`** — complete methodology, experiments, and conclusions
- **`Final Presentation.pptx`** — summary slides of the project

## 👤 Author

**Pavlos** ([@Pavlos43](https://github.com/Pavlos43))

## 📄 License

This project was created for academic purposes as part of a university course.
