"""
Hand_Evaluator.py
----------------------------------------------------------------------------------
Utility file that provides player hand evaluation in Texas Hold'em.

Card input formats (all auto-detected):
    Objects with .rank (int 2-14 OR str '2'-'A') and .suit (str 'S'/'H'/'D'/'C')
    'Ah', 'Tc', 'Kd', '2s'   rank-char first, lower/upper suit  (standard)
    'SA', 'H2', 'DT', 'CK'   suit-char first, upper             (RLCard format)
    'AH', 'TC', '2S', 'KD'   rank-char first, upper suit

Street is inferred from len(community_cards):
    0  → preflop   (preflop hand category used)
    ≥1 → postflop  (postflop hand category used)

Probability tables are renormalised over legal actions only.
"""

from collections import Counter
from itertools import combinations

# ================================================================================ #
# HAND EVALUATION  (opponent decides which action to take based on card strength)  #
# ================================================================================ #

def score_5(cards) -> tuple:
    """
    Return a comparable score tuple for exactly 5 Card objects.

    Tuple element [0] is the hand type (higher = better):
        8  straight flush   |  7  four of a kind    |  6  full house
        5  flush            |  4  straight          |  3  three of a kind
        2  two pair         |  1  one pair          |  0  high card

    All remaining elements are tiebreaker ranks (higher = better).
    """
    ranks = sorted((c.rank for c in cards), reverse=True)
    suits = [c.suit for c in cards]

    is_flush = (len(set(suits)) == 1)       # all cards are of the same suit

    # Straight detection requires exactly 5 distinct ranks
    u = sorted(set(ranks), reverse=True)
    (is_str, s_hi) = (False, 0)
    if len(u) == 5:
        if u[0] - u[4] == 4:                # sorted cards => if difference between lowest and highest is 4, then they have values i, i+1, i+2 etc.
            (is_str, s_hi) = (True, u[0])   # normal straight
        elif set(u) == {14, 5, 4, 3, 2}:
            (is_str, s_hi) = (True, 5)      # wheel  A-2-3-4-5

    # Group ranks by frequency (freq desc, rank desc)
    cnt = Counter(ranks)
    grp = sorted(cnt, key=lambda r: (cnt[r], r), reverse=True)
    frq = [cnt[g] for g in grp]

    if is_flush and is_str:           return (8, s_hi)                            # royal flush is also included here since anything equal or better is considered a "monster" hand
    elif frq[0] == 4:                 return (7, grp[0], grp[1])                  # 4 of a kind => 4 cards with the same rank       =>  return the rank of the 4 same cards + the 5th card
    elif frq[0] == 3 and frq[1] == 2: return (6, grp[0], grp[1])                  # full house => 3 of a kind + pair                =>  return the rank of the 3 same cards + the pair cards
    elif is_flush:                    return (5, *ranks)                          # flush => all cards are of the same suit         =>  return the rank of all 5 cards 
    elif is_str:                      return (4, s_hi)                            # straight => cards with ranks i, i+1, ... , i+4  =>  return (4, highest_rank) [straight flush would be (8, highest rank)]
    elif frq[0] == 3:                 return (3, grp[0], grp[1], grp[2])          # 3 of a kind => 3 cards with the same rank       =>  return the rank of the 3 same cards + the 4th and 5th cards
    elif frq[0] == 2 and frq[1] == 2: return (2, grp[0], grp[1], grp[2])          # 2-pair => self explanatory                      =>  return the rank of the 4 pair cards (2x2) + the 5th card
    elif frq[0] == 2:                 return (1, grp[0], grp[1], grp[2], grp[3])  # pair => self explanatory                        =>  return the rank of the pair cards + the other 3 cards
    return (0, *ranks)                                                            # high card => self explanatory                   =>  return the rank of all 5 cards


def best_hand(cards) -> tuple:
    """
    Best-5-card score from a list of 5-7 Card objects.
    Compare the quality of all 5-card combos (out of 5-7 cards)
    and return the score_5 tuple for the highest quality combination.

    Example:
        Cards: ('J','Q') + ('K','T','A','9')
        Combos:
            'T-J-Q-K-A' => straight  (4, 14)
            '9-T-J-Q-K' => straight  (4, 13)
            '9-J-Q-K-A' => high-card (0, 9, 11, 12, 13, 14)
            '9-T-J-Q-A' => high-card (0, 9, 10, 11, 12, 14)
            '9-T-J-K-A' => high-card (0, 9, 10, 11, 13, 14)
            '9-T-Q-K-A' => high-card (0, 9, 10, 12, 13, 14)
        Returns (4,14).
    """
    card_list = list(cards)
    if len(card_list) == 5:                                                       # 2 private cards + 3 public cards = 1st post-flop betting round
        return score_5(card_list)
    return max(score_5(combo) for combo in combinations(card_list, 5))            # 2nd and 3rd post-flop betting rounds


def hand_type(cards) -> int:
    """Integer hand type 0-8 for the best 5-card hand from 5-7 cards."""
    return best_hand(cards)[0]


# ====================== #
# PREFLOP CLASSIFICATION #
# ====================== #

def classify_preflop(hole: list) -> str:
    """
    Classify a 2-card preflop holding.
    Returns 'premium', 'strong', 'marginal', or 'weak'.
    
    Classes:
        'premium'  - 'AA', 'KK', 'QQ', 'JJ', 'TT'   |  'AKs', 'AKo', 'AQs'
        'strong'   - '99', '88', '77'               |  'AQo', 'AJs', 'AJo', 'ATs'   |  'KQs', 'KQo', 'KJs'  |  'QJs'
        'marginal' - '66', '55', '44', '33', '22'   |  'A2s'-'A9s'                  |  'KJo', 'KTs'         |  'QJo', 'QTs'  |  'JTs', 'T9s', '98s', '87s'
        'weak'     - everything else
    """
    (r1, r2) = sorted((hole[0].rank, hole[1].rank), reverse=True)  # r1 ≥ r2
    suited = (hole[0].suit == hole[1].suit)
    pair = (r1 == r2)

    # --------------------------- Pairs --------------------------------
    if pair:
        if r1 >= 10: return 'premium'   # 'AA', 'KK', 'QQ', 'JJ', 'TT'
        if r1 >=  7: return 'strong'    # '99', '88', '77'
        return 'marginal'               # '66', ... , '22'

    # ---------------------- Ace-high hands ----------------------------
    if r1 == 14:
        if r2 == 13:               return 'premium'    # AKs / AKo
        elif r2 == 12 and suited:  return 'premium'    # AQs
        elif r2 == 12:             return 'strong'     # AQo
        elif r2 == 11:             return 'strong'     # AJs / AJo
        elif r2 == 10 and suited:  return 'strong'     # ATs
        elif suited:               return 'marginal'   # A2s – A9s
        return 'weak'                                  # ATo / A2o – A9o

    # ------------------------ King-high -------------------------------
    elif r1 == 13:
        if r2 == 12:               return 'strong'     # KQs / KQo
        elif r2 == 11 and suited:  return 'strong'     # KJs
        elif r2 == 11:             return 'marginal'   # KJo
        elif r2 == 10 and suited:  return 'marginal'   # KTs
        return 'weak'

    # ----------------------- Queen-high -------------------------------
    elif r1 == 12:
        if r2 == 11 and suited:    return 'strong'     # QJs
        elif r2 == 11:             return 'marginal'   # QJo
        elif r2 == 10 and suited:  return 'marginal'   # QTs
        return 'weak'

    # ---------- Suited connectors (Jack-high and below) ---------------
    elif r1 == 11 and r2 == 10 and suited: return 'marginal'   # JTs
    elif r1 == 10 and r2 ==  9 and suited: return 'marginal'   # T9s
    elif r1 ==  9 and r2 ==  8 and suited: return 'marginal'   # 98s
    elif r1 ==  8 and r2 ==  7 and suited: return 'marginal'   # 87s

    # else
    return 'weak'


# ======================= #
# POSTFLOP CLASSIFICATION #
# ======================= #

def almost_flush(cards) -> bool:
    """True if any suit appears ≥ 4 times among the given cards."""
    return max(Counter(c.suit for c in cards).values()) >= 4


def almost_straight(cards) -> bool:
    """
    True if ≥ 4 distinct card ranks fit inside any 5-consecutive-rank window
    (including the wheel window A-2-3-4-5).
    """
    ranks = {c.rank for c in cards}
    for low in range(2, 11):                            # windows 2–6 ... 10–14 (T–A)
        if len(ranks & set(range(low, low + 5))) >= 4:
            return True
    return len(ranks & {14, 2, 3, 4, 5}) >= 4           # wheel window


def classify_postflop(hole: list, comm: list) -> str:
    """
    Classify a postflop holding.

    Parameters
    ----------
    hole : list of exactly 2 Card objects (private cards)
    comm : list of 3-5 Card objects (community cards)

    Returns
    ----------
    'monster' : 3 of a kind or better; straight or better
    'strong'  : 2-pair;
                pocket pair whose rank exceeds every board rank (overpair);
                one hole card pairs the top board rank AND the other hole card ≥ J
    'medium'  : any other made pair
    'draw'    : no pair, but ≥ 4 cards share a suit OR ≥ 4 ranks fit a straight window
    'weak'    : all other hands
    """
    all_cards = hole + comm
    ht = hand_type(all_cards)

    # --- 3 of a kind (level 3) / Straight (level 4) or better: Monster ---
    if ht >= 3:
        return 'monster'

    # --------------------- 2-pair (level 2): Strong ----------------------
    if ht == 2:
        return 'strong'

    # -------------- Single pair (level 1): Strong or Medium --------------
    if ht == 1:
        (h0, h1)   = (hole[0].rank, hole[1].rank)
        comm_ranks = [c.rank for c in comm]
        top_card   = max(comm_ranks) if comm_ranks else 0

        # Overpair: private cards form a pocket pair above every board card
        if h0 == h1 and h0 > top_card:
            return 'strong'

        # Top pair with J+ kicker:
        #   one hole card pairs the highest board rank,
        #   the other hole card is J (rank 11) or better
        if comm_ranks:
            for (my_rank, kicker) in ((h0, h1), (h1, h0)):
                if my_rank == top_card and kicker >= 11:
                    return 'strong'

        return 'medium'

    # ------------------ No pair (level 0): Draw or Weak ------------------
    if almost_flush(all_cards) or almost_straight(all_cards):
        return 'draw'

    return 'weak'