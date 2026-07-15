"""
Opponent_Baselines.py
----------------------------------------------------------------------------
Fixed-strategy baseline agents for heads-up limit Texas Hold'em.

Each agent exposes:
    act(hole_cards, community_cards, legal_actions) → str
        hole_cards      - iterable of 2 cards
        community_cards - iterable of 0-5 cards (empty preflop)
        legal_actions   - list of legal action strings from the environment
                        (subset of {'fold', 'check', 'call', 'raise'})
        returns one element of legal_actions

Probability tables are renormalised over legal actions only.
For Tight and Aggressive, 'call' is transparently replaced by 'check'
whenever 'check' is a legal action (spec rule).
"""

import random as rnd
import Card
import Hand_Evaluator

# ============================================================ #
# PROBABILITY TABLES  (module-level constants, never mutated)  #
# ============================================================ #

# 'call' entries are converted to 'check' by decide() when check is legal.
# Raise probabilities of 0.00 are omitted (they have no effect after renorm).

TIGHT_PREFLOP = {
    'premium':  {'raise': 0.70, 'call': 0.25, 'fold': 0.05},
    'strong':   {'raise': 0.35, 'call': 0.55, 'fold': 0.10},
    'marginal': {'raise': 0.05, 'call': 0.45, 'fold': 0.50},
    'weak':     {               'call': 0.15, 'fold': 0.85},
}

TIGHT_POSTFLOP = {
    'monster': {'raise': 0.70, 'call': 0.30},
    'strong':  {'raise': 0.40, 'call': 0.55, 'fold': 0.05},
    'medium':  {'raise': 0.10, 'call': 0.65, 'fold': 0.25},
    'draw':    {'raise': 0.10, 'call': 0.55, 'fold': 0.35},
    'weak':    {               'call': 0.20, 'fold': 0.80},
}

AGG_PREFLOP = {
    'premium':  {'raise': 0.90, 'call': 0.10},
    'strong':   {'raise': 0.75, 'call': 0.20, 'fold': 0.05},
    'marginal': {'raise': 0.45, 'call': 0.40, 'fold': 0.15},
    'weak':     {'raise': 0.25, 'call': 0.35, 'fold': 0.40},
}

AGG_POSTFLOP = {
    'monster': {'raise': 0.90, 'call': 0.10},
    'strong':  {'raise': 0.75, 'call': 0.20, 'fold': 0.05},
    'medium':  {'raise': 0.45, 'call': 0.40, 'fold': 0.15},
    'draw':    {'raise': 0.50, 'call': 0.35, 'fold': 0.15},
    'weak':    {'raise': 0.25, 'call': 0.35, 'fold': 0.40},
}


class Opponent:
    """
    Fixed-strategy poker opponent.

    Styles:
        'Random' - Chooses uniformly at random among the legal actions.
        'Passive'- Checks or calls almost all the time, rarely raises, rarely folds. Decision depends on the action availability, not on hand strength.
        'Tight'  - Selective, conservative player. Plays few hands and rarely bluffs. Folds frequently without a strong holding; value-bets monster hands.
        'Aggressive' - Bets and raises frequently with all hand strengths, maximising pressure on the opponent at every decision point.
    """

    def __init__(self, style: str):
        self.style = style

    def reset(self):
        """No internal state to reset — baselines are stateless. Here for API compatibility."""
        pass
    # ================================================================================= #
    # ACTION DECISION (opponent decides to take a legal action based on probabilities)  #
    # ================================================================================= #

    def decide(self, dist: dict, legal: set) -> str:
        """
        Draw an action from a probability distribution, restricted to legal actions.

        Steps:\n
        1. If 'check' is in legal and 'call' appears in dist, redirect call's
        probability mass to 'check' (Tight / Aggressive rule from the spec).
        2. Remove every action that is either illegal or has zero probability.
        3. Renormalise and sample.
        4. Fall back to uniform over legal if nothing survives (edge case).
        """
        prob_distribution = dict(dist)  # shallow copy to avoid mutating the module-level table

        # If 'check' is legal, use 'check' instead of 'call'
        if 'check' in legal and 'call' in prob_distribution:
            prob_distribution['check'] = prob_distribution.pop('call')

        filtered = {action: prob for (action, prob) in prob_distribution.items() if (action in legal and prob > 0)}
        if not filtered:
            return rnd.choice(list(legal))

        total = sum(filtered.values())     # Remaining action probabilities p1 + p2 + ... add up to k <= 1
        r = rnd.random() * total           # Normalized random number [0,1] -> [0,k]
        cumulative = 0
        for action, prob in filtered.items():
            cumulative += prob             # Initially, cumulative = p1, if action 1 isn't chosen (r > p1), cumulative = p1 + p2 etc.
            if r < cumulative:
                return action
        return next(reversed(filtered))    # floating-point safety net


    def act(self, hole_cards, community_cards, legal_actions: list) -> str:
        """
        Evaluate hand and decide which action to take among the list of legal actions.\n
        Depending on opponent style, the action-taking probability distribution will be different.
        """

        # ------------------------- Random opponent strategy -------------------------
        if self.style == 'Random':
            return rnd.choice(legal_actions)

        # ------------------------ Limping opponent strategy -------------------------
        if self.style == 'Limping':
            legal = set(legal_actions)
            if 'check' in legal:
                dist = {'check': 0.85, 'raise': 0.15}
            elif 'raise' in legal:
                dist = {'call': 0.90, 'raise': 0.05, 'fold': 0.05}
            else:
                dist = {'call': 0.95, 'fold': 0.05}

            return self.decide(dist, legal)

        # ------------------------- Tight opponent strategy --------------------------
        elif self.style == 'Tight':
            hole  = Card.parse_all(hole_cards)
            comm  = Card.parse_all(community_cards)
            legal = set(legal_actions)

            if not comm:                                  # preflop
                hand_quality  = Hand_Evaluator.classify_preflop(hole)
                dist = TIGHT_PREFLOP[hand_quality]
            else:                                         # postflop
                hand_quality  = Hand_Evaluator.classify_postflop(hole, comm)
                dist = TIGHT_POSTFLOP[hand_quality]

            return self.decide(dict(dist), legal)
        
        # ---------------------- Aggressive opponent strategy ------------------------
        elif self.style == 'Aggressive':
            hole  = Card.parse_all(hole_cards)
            comm  = Card.parse_all(community_cards)
            legal = set(legal_actions)

            if not comm:
                hand_quality  = Hand_Evaluator.classify_preflop(hole)
                dist = AGG_PREFLOP[hand_quality]
            else:
                hand_quality  = Hand_Evaluator.classify_postflop(hole, comm)
                dist = AGG_POSTFLOP[hand_quality]

            return self.decide(dict(dist), legal)