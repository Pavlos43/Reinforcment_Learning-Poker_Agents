"""
DQN_Agent.py
------------------------------------------------------------------------------
Deep Q-Network agent for heads-up Limit Texas Hold'em.

Components
----------
QNetwork     – MLP that maps state → Q-values for each action
ReplayBuffer – Fixed-size circular buffer of (s, a, r, s', done) tuples
DQNAgent     – Ties everything together: ε-greedy policy, learning, target net

Algorithm
---------
Standard DQN with two improvements that matter a lot for poker:
1. Target network  – frozen copy of Q-net, synced every `target_update` steps.
                        Stabilises the Bellman targets.
2. Illegal action masking – Q-values for illegal actions are set to -inf
                        before both action selection AND target computation.
                        Without this the agent wastes thousands of steps
                        unlearning illegal moves.

Hyperparameter guide
--------------------
lr              = 1e-4    start here; 1e-3 can diverge, 1e-5 is too slow
gamma           = 0.99    standard; poker hands are short so this is fine
batch_size      = 128     bigger = more stable gradients, more memory
buffer_capacity = 100_000 keep at least 50k; poker rewards are sparse
epsilon_start   = 1.0     always start fully random
epsilon_end     = 0.05    keep some exploration; poker has high variance
epsilon_decay   = 0.9995  over ~13k steps from 1.0 → 0.05; tune to your
                            episode length (decay slower for longer training)
target_update   = 1_000   sync target net every 1000 *steps* (not episodes)
min_buffer      = 1_000   don't start learning until buffer has this many
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QNetwork(nn.Module):
    """
    Three-layer MLP: state_dim → 256 → 256 → num_actions.

    ReLU activations, no output activation (raw Q-values).
    Batch norm is intentionally omitted — it interacts badly with the
    target network when batch statistics drift between updates.
    """

    def __init__(self, state_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden),nn.ReLU(),nn.Linear(hidden, hidden),nn.ReLU(),nn.Linear(hidden, num_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ------------------------------------------------------------------ #
#  Replay Buffer                                                       #
# ------------------------------------------------------------------ #

class ReplayBuffer:
    """
    Circular buffer storing (state, action, reward, next_state, done) tuples.

    Uses collections.deque with maxlen so old transitions are automatically
    discarded once capacity is reached.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_legal_mask):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
            np.array(next_legal_mask, dtype=bool),
            ))

    def sample(self, batch_size: int):
        """
        Sample a random minibatch.

        Returns five numpy arrays:
            states      (B, state_dim)
            actions     (B,)
            rewards     (B,)
            next_states (B, state_dim)
            dones       (B,)  float32, 1.0 = terminal
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(next_masks,  dtype=bool),       
        )

    def __len__(self):
        return len(self.buffer)
# ------------------------------------------------------------------ #
#  DQN Agent                                                           #
# ------------------------------------------------------------------ #

class DQNAgent:
    """
    DQN agent with target network and illegal-action masking.

    Parameters
    ----------
    state_dim       : int   – length of the state vector (72 for limit-holdem)
    num_actions     : int   – number of actions (4 for limit-holdem)
    lr              : float – Adam learning rate
    gamma           : float – discount factor
    epsilon_start   : float – initial exploration rate
    epsilon_end     : float – minimum exploration rate
    epsilon_decay   : float – multiplicative decay applied each step
    batch_size      : int   – minibatch size for each gradient update
    buffer_capacity : int   – maximum replay buffer size
    target_update   : int   – how many steps between target net syncs
    min_buffer      : int   – minimum buffer size before learning starts
    hidden          : int   – hidden layer width for QNetwork
    """

    def __init__(
        self,
        state_dim:       int   = 72,
        num_actions:     int   = 4,
        lr:              float = 1e-4,
        gamma:           float = 0.99,
        epsilon_start:   float = 1.0,
        epsilon_end:     float = 0.05,
        epsilon_decay:   float = 0.9995,
        batch_size:      int   = 128,
        buffer_capacity: int   = 100_000,
        target_update:   int   = 1_000,
        min_buffer:      int   = 1_000,
        hidden:          int   = 256,
    ):
        self.num_actions  = num_actions
        self.gamma        = gamma
        self.epsilon      = epsilon_start
        self.epsilon_end  = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size   = batch_size
        self.target_update = target_update
        self.min_buffer   = min_buffer

        # Networks
        self.online_net = QNetwork(state_dim, num_actions, hidden).to(DEVICE)
        self.target_net = QNetwork(state_dim, num_actions, hidden).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()   # target net is never trained directly

        # Optimiser & loss
        self.optimiser = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)

        # Step counter (used for target network sync)
        self.steps = 0

    # -------------------------------------------------------------- #
    #  Action selection                                                #
    # -------------------------------------------------------------- #

    def act(self, state: np.ndarray, legal_mask: np.ndarray) -> int:
        """
        ε-greedy action selection with illegal-action masking.

        Parameters
        ----------
        state      : np.ndarray (state_dim,)
        legal_mask : np.ndarray (num_actions,) bool – True = legal

        Returns
        -------
        action : int
        """
        if random.random() < self.epsilon:
            # Random choice among LEGAL actions only
            legal_actions = np.where(legal_mask)[0]
            return int(np.random.choice(legal_actions))

        # Greedy: mask illegal actions then take argmax
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = self.online_net(state_t).squeeze(0).cpu().numpy()

        q_values[~legal_mask] = -np.inf   # illegal actions can never win
        return int(np.argmax(q_values))

    # -------------------------------------------------------------- #
    #  Storing transitions                                             #
    # -------------------------------------------------------------- #
    def store(self, state, action: int, reward: float, next_state, done: bool, next_legal_mask):
        self.buffer.push(state, action, reward, next_state, done, next_legal_mask)

    def learn(self) -> float | None:
        """
        Sample a minibatch and perform one gradient update.

        Returns the loss value (float) or None if the buffer isn't
        large enough yet.
        """
        if len(self.buffer) < self.min_buffer:
            return None

        self.steps += 1

        # ── Sample ──────────────────────────────────────────────────
        states, actions, rewards, next_states, dones, next_masks = self.buffer.sample(self.batch_size)

        states_t      = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        actions_t     = torch.tensor(actions, dtype=torch.int64  ).to(DEVICE)
        rewards_t     = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        dones_t       = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
        next_masks_t   = torch.tensor(next_masks,  dtype=torch.bool   ).to(DEVICE)  # ← new

        # ── Current Q-values for the actions actually taken ─────────
        q_current = self.online_net(states_t)                        # (B, num_actions)
        q_current = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

        # ── Target Q-values (Bellman) ────────────────────────────────
        with torch.no_grad():
            q_next = self.target_net(next_states_t) 
            q_next[~next_masks_t] = -float('inf')          
            q_next_max = q_next.max(dim=1).values                    # (B,)
            targets = rewards_t + self.gamma * q_next_max * (1 - dones_t)

        # ── Gradient update ──────────────────────────────────────────
        loss = self.loss_fn(q_current, targets)
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        #self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    # -------------------------------------------------------------- #
    #  RLCard-compatible interface (needed for PortableRLCardAgent)   #
    # -------------------------------------------------------------- #

    def eval_step(self, state: dict):
        """
        RLCard-compatible interface used by PortableRLCardAgent.

        Parameters
        ----------
        state : dict  – raw RLCard state dictionary with keys:
                        'obs'          np.ndarray (72,)
                        'legal_actions' dict {int: None}

        Returns
        -------
        (action, info) : (int, dict)
        """
        obs_vec    = state['obs'].astype(np.float32)          # (72,)
        legal_ints = list(state.get('legal_actions', {}).keys())

        legal_mask = np.zeros(self.num_actions, dtype=bool)
        for a in legal_ints:
            if 0 <= a < self.num_actions:
                legal_mask[a] = True

        # Use greedy policy (no exploration) during eval
        state_t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.online_net(state_t).squeeze(0).cpu().numpy()

        q_values[~legal_mask] = -np.inf
        action = int(np.argmax(q_values))
        return action, {}

    def step(self, state: dict) -> int:
        """Alias for eval_step that returns only the action integer."""
        action, _ = self.eval_step(state)
        return action

    def save(self, path: str):
        """Save the online network weights to disk."""
        torch.save({
            'online_net':  self.online_net.state_dict(),
            'target_net':  self.target_net.state_dict(),
            'optimiser':   self.optimiser.state_dict(),
            'epsilon':     self.epsilon,
            'steps':       self.steps,
        }, path)

    def load(self, path: str):
        """Load a checkpoint saved by save()."""
        ckpt = torch.load(path, map_location=DEVICE)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimiser.load_state_dict(ckpt['optimiser'])
        self.epsilon = ckpt['epsilon']
        self.steps   = ckpt['steps']