"""
Environment.py
------------------------------------------------------------------------------
A clean wrapper around RLCard's limit-holdem environment for the DQN project.

Responsibilities
----------------
1. Translate RLCard's raw obs dict into numpy arrays your DQN agent can use.
2. Handle the opponent's turn automatically, so the training loop only ever
    sees the DQN agent's perspective (single-agent interface).
3. Expose a simple  reset() / step(action)  API.
4. Bridge between your Opponent_Baselines agents and RLCard's integer actions.

RLCard quick-reference (limit-holdem)
--------------------------------------
State vector  : 72 floats
    [0:52]  – one-hot card encoding (52 cards, index = rank*4 + suit_order)
    [52:72] – betting history (4 rounds × 5 slots each, values 0/1)

Actions (integers):
    0 – call
    1 – raise
    2 – fold
    3 – check

Reward: 0 at every intermediate step; +X / -X (in big blinds) at the hand end.

num_players : 2  (heads-up)
DQN agent   : always player 0
Opponent    : always player 1
"""

import numpy as np
import rlcard
from Opponent_Baselines import Opponent


STATE_DIM  = 72   # length of obs['obs']
NUM_ACTIONS = 4   # fold / check / call / raise
INT_TO_STR = {0: 'call', 1: 'raise', 2: 'fold', 3: 'check'}
STR_TO_INT = {'call': 0, 'raise': 1, 'fold': 2, 'check': 3}

class LimitHoldemEnv:
    """
    Single-agent wrapper around RLCard's limit-holdem environment.

    The DQN agent always plays as player 0.
    The opponent (an Opponent_Baselines agent) plays as player 1 and its
    turns are handled automatically inside step().

    Parameters
    ----------
    opponent_style : str
        One of 'Random', 'Limping', 'Tight', 'Aggressive'.

    Usage
    -----
        env = LimitHoldemEnv(opponent_style='Random')

        state, legal_mask = env.reset()
        done = False
        while not done:
            action = agent.act(state, legal_mask)   # integer 0-3
            next_state, reward, done, legal_mask = env.step(action)
    """

    def __init__(self, opponent_style):
        self._env      = rlcard.make('limit-holdem')
        self._opponent = Opponent(style=opponent_style)
        self.state_dim  = STATE_DIM
        self.num_actions = NUM_ACTIONS

    def reset(self):
        """
        Start a new hand.

        Returns
        -------
        state      : np.ndarray, shape (72,), dtype float32
        legal_mask : np.ndarray, shape (4,),  dtype bool
            True at index i means action i is currently legal.
        """
        self._opponent.reset()
        obs, player_id = self._env.reset()

        while player_id != 0:
            result = self._opponent_step(obs)
            if len(result) == 3:
                # Hand ended before agent got to act — start a new hand
                self._opponent.reset()
                obs, player_id = self._env.reset()
            else:
                obs, player_id = result
        return self._parse_obs(obs)

    def step(self, action: int):
        """
        Take one action as the DQN agent (player 0).

        Parameters
        ----------
        action : int  (0=call, 1=raise, 2=fold, 3=check)

        Returns
        -------
        next_state : np.ndarray, shape (72,), dtype float32
                    All zeros if the hand ended (check `done`).
        reward     : float   – final payoff in big blinds, 0 mid-hand.
        done       : bool    – True when the hand is over.
        legal_mask : np.ndarray, shape (4,), dtype bool
                    Meaningful only when done=False.
        """
        obs, reward, done = self._dqn_step(action)
        if done:
            return np.zeros(STATE_DIM, dtype=np.float32), float(reward[0]), True, np.ones(NUM_ACTIONS, dtype=bool)

        # Opponent takes turns until it is the DQN agent's move again
        player_id = self._env.get_player_id()        
        while player_id != 0:
            result = self._opponent_step(obs)
            if isinstance(result, tuple) and len(result) == 3:
                # Hand ended during opponent's turn
                obs_dict, reward, done = result
                return np.zeros(STATE_DIM, dtype=np.float32), float(reward[0]), True, np.ones(NUM_ACTIONS, dtype=bool)
            obs, player_id = result

        state, legal_mask = self._parse_obs(obs)
        return state, 0.0, False, legal_mask

    def _parse_obs(self, obs: dict):
        """Convert a raw RLCard obs dict to (state_vector, legal_mask)."""
        state = obs['obs'].astype(np.float32)               # shape (72,)
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for a in obs.get('legal_actions', {}):
            mask[a] = True
        return state, mask

    def _dqn_step(self, action: int):
        """
        Submit the DQN agent's action to RLCard.
        Returns (obs_or_trajectories, reward_or_None, done).
        """
        next_obs, next_player = self._env.step(action)
        if self._env.is_over():
            return None, self._env.get_payoffs(), True
        return next_obs, None, False

    def _opponent_step(self, obs: dict):
        """
        Let the Opponent_Baselines agent pick and submit its action.

        Returns either:
            (obs_dict, player_id)   – if the hand continues, or
            (None, payoffs, True)   – if the hand ended after this action.
        """
        raw = obs.get('raw_obs', {})
        hole_cards  = raw.get('hand', [])
        comm_cards  = raw.get('public_cards', [])

        # Build legal action strings from the integer legal_actions dict
        legal_ints    = list(obs.get('legal_actions', {}).keys())
        legal_strings = [INT_TO_STR[a] for a in legal_ints if a in INT_TO_STR]

        chosen_str = self._opponent.act(hole_cards, comm_cards, legal_strings)
        chosen_int = STR_TO_INT.get(chosen_str, 1)

        next_obs, next_player = self._env.step(chosen_int)

        if self._env.is_over():
            return None, self._env.get_payoffs(), True
        return next_obs, next_player