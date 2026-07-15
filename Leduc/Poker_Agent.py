import numpy as np

class QLearningPlayer:
    """Simplified Q-Learning agent for LeDuc Poker"""
    
    def __init__(self, model_path, alpha, gamma, epsilon):
        # Q-learning parameters
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.bet = 0
        self.raised = False

        # Simplified state space (round is implicit from public_card):
        # - Private card: J, Q, K (3)
        # - Public card: none, J, Q, K (4)
        # - Action history: None, 'Check/Call', 'Raise', 'Check/Call + Raise', 'Raise + Raise', 'Check/Call + Raise + Raise' (6)
        # - Opponent style: 'Pal-Bot', 'Random', 'Tight', 'Limping', 'Aggressive', 'Honest-strength' (6)
        # - Position: first or second (2)

        self.state_size = 3 * 4 * 6 * 2     # 144 states
        self.action_size = 4  # check, call, raise, fold
        
        # Initialize or load Q-table (144 × 4 = 576 values)
        if model_path!=None:
            self.Q = self.load_model(model_path)
        else:
            self.Q = np.random.rand(self.state_size, self.action_size) * 0.001
        
        # Matrix holding the times each q-value was updated in learn() => used for alpha decaying
        self.N = np.zeros_like(self.Q)

        # Tracking for learning
        self.last_state = None
        self.last_action = None
        #self.last_legal_actions = None
        self.last_private_card = None
        self.last_play_first = None
        self.last_public_card = None
        
        # Dictionaries for mapping 
        self.card_map = {'J': 0, 'Q': 1, 'K': 2}
        self.history_map = {'':0, 'c':1, 'r':2, 'cr':3, 'rr':4, 'crr':5}
        self.blind_map = {'SB': 0, 'BB': 1}                                         # SB: Small blind, BB: Big blind (SB plays first)
        self.action_to_idx_map = {'check': 0, 'call': 1, 'raise': 2, 'fold': 3}
        self.idx_to_action_map = {0: 'check', 1: 'call', 2: 'raise', 3: 'fold'}
        self.reset()
        
    # Reset tracking for new game
    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_legal_actions = None
        self.last_private_card = None
        self.last_play_first = None
        self.last_public_card = None
        self.bet = 0
        self.raised = False


    # Save Q-table to file
    def save_model(self, path):
        np.savetxt(path, self.Q, delimiter=',')
        print(f"Model saved to {path}")
    

    # Load Q-table from file
    def load_model(self, path):
        print(f"Model loaded from {path}")
        return np.loadtxt(path, delimiter=',')


    def encode_state(self, private_card, public_card, history, blind):
        """
        Convert game state to a unique state ID (0 to 143)
        
        Args:
            private_card: 'J', 'Q', or 'K'
            public_card: None (round 1) or 'J','Q','K' (round 2)
            history: 'c', 'r', 'cr', 'rr' etc.
            blind: 'SB'/'BB' => also indicates who plays first
        """

        # Private card (0-2)
        private_idx = self.card_map[private_card]
        
        # Public card (0-3: 0=none, 1=J, 2=Q, 3=K)
        if public_card is None:
            public_idx = 0
        else:
            public_idx = self.card_map[public_card] + 1
        
        # Action history (0-5)
        history_idx = self.history_map[history]

        # Who plays first (Big-blind, Small-blind)
        blind_idx = self.blind_map[blind]
    
        # Calculate flattened state index
        state =  (private_idx * 4 * 6 * 2) + (public_idx * 6 * 2) + (history_idx * 2) + blind_idx
        return state
    

    def choose_action(self, state, legal_actions):
        """
        Epsilon-greedy action selection.
        Only considers LEGAL actions - no need to encode legality in state!
        """
        
        # Explore a random legal action with probability epsilon
        if np.random.random() < self.epsilon:
            return np.random.choice(legal_actions)

        # Choose the action that you think is optimal with probability 1-epsilon
        legal_indices = [self.action_to_idx_map[a] for a in legal_actions]
        q_values = [self.Q[state, idx] for idx in legal_indices]
        best_q = max(q_values)
        best_indices = [legal_indices[i] for i, q in enumerate(q_values) if q == best_q]
        return self.idx_to_action_map[np.random.choice(best_indices)]
    

    def action_1(self, private_card, legal_actions, history, player_starts):
        """
        First betting round (pre-flop)
        
        Args:
            private_card: 'J', 'Q', or 'K'
            legal_actions: list of allowed actions (e.g., ['check', 'raise'])
            history: betting history (e.g., 'c', 'r', 'cr', 'rr')
            player_starts: 'player' if the player starts, 'opponent' otherwise
        """
        # Encode current state ("public_card = None" indicates round 1)
        b = 'SB' if (player_starts == "player") else 'BB'
        state = self.encode_state(private_card = private_card, public_card = None, history = history, blind = b)
        
        # Choose action
        action = self.choose_action(state, legal_actions)
        
        # Store for learning
        self.last_state = state
        self.last_action = action
        self.next_state = None
        
        return action
    

    def action_2(self, private_card, legal_actions, history, player_starts, public_card):
        """
        Second betting round (post-flop)
        
        Args:
            private_card: 'J', 'Q', or 'K'
            legal_actions: list of allowed actions
            history: betting history (e.g., 'c', 'r', 'cr', 'rr')
            player_starts: 'player' if the player starts, 'opponent' otherwise
            public_card: 'J', 'Q', or 'K' (the card on the table)
        """
        # Encode current state
        b = 'SB' if (player_starts == "player") else 'BB'
        state = self.encode_state(private_card = private_card, public_card = public_card, history = history, blind = b)
        
        # Choose action
        action = self.choose_action(state, legal_actions)
        
        # Store for learning
        self.last_state = state
        self.last_action = action
        self.next_state = None
        
        return action
    

    def learn(self, state, action, reward, next_state, done = False):
        """
        Update Q-values after receiving reward
        
        Args:
            reward: chip change from this hand/action
            next_state: encoded state for the next step
            done: whether the episode is over
        """
        if action is None:
            return False
        
        action = self.action_to_idx_map[action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * target
        self.N[state, action] += 1

        #alpha becomes smaller for state-action pairs with a large amount of updates => eventual lock-in
        self.alpha = max(0.01, 20/(20+self.N[state, action])) 
        
        # It would be wrong to place the epsilon decay here. Decaying should happen between training episodes
        # self.epsilon = max(0.01, 0.999*self.epsilon)          -- epsilon drops with rate e(t) = 0.999^t

        return False