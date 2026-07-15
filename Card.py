"""
Card.py
-------------------------------------
Internal Card Object Representation.
"""

RANKS    = '23456789TJQKA'                            # 'T' = 10
RANK_VAL = {r: i + 2 for i, r in enumerate(RANKS)}    # '2'→2, …, 'A'→14
RANK_CH  = {v: r for r, v in RANK_VAL.items()}        #  2→'2', …, 14→'A'
SUIT_NORM = {
    's': 's', 'S': 's', '♠': 's',
    'h': 'h', 'H': 'h', '♥': 'h',
    'd': 'd', 'D': 'd', '♦': 'd',
    'c': 'c', 'C': 'c', '♣': 'c',
}
RANK_CHARS = frozenset(RANKS)           # characters that start a rank
SUIT_CHARS = frozenset('SHDCshdc♠♥♦♣')  # characters that start a suit


class Card:
    """Minimal immutable card used internally by every baseline."""

    __slots__ = ('rank', 'suit')

    def __init__(self, rank_int: int, suit_char: str):
        self.rank = rank_int   # int  2–14
        self.suit = suit_char  # str  'c'|'d'|'h'|'s'

    def __repr__(self):
        return f"{RANK_CH[self.rank]}{self.suit}"


def parse(card) -> Card:
    """
    Convert any card representation into a Card.

    Accepted inputs
    ---------------
    Card                           already parsed; returned as-is
    object with .rank and .suit    RLCard Card objects and similar
    'Ah', 'Tc', '2d', 'Ks'         rank-char first, any suit case
    'SA', 'H2', 'DT', 'CK'         suit-char first  (RLCard string format)
    'AH', 'TC', 'KD', '2S'         rank-char first, upper suit
    """
    if isinstance(card, Card):
        return card

    # ----------- Object with .rank / .suit (RLCard Card, etc.) ------------
    if hasattr(card, 'rank') and hasattr(card, 'suit'):
        r = card.rank
        s = card.suit
        if isinstance(r, str):
            r = RANK_VAL.get(r.upper())
            if r is None:
                raise ValueError(f"Unrecognised rank in card object: {card!r}")
        s = SUIT_NORM.get(str(s)[0] if s else '', None)
        if s is None:
            raise ValueError(f"Unrecognised suit in card object: {card!r}")
        return Card(int(r), s)

    # ------------------------------- String -------------------------------
    raw = str(card).strip()
    if len(raw) < 2:
        raise ValueError(f"Cannot parse card: {card!r}")

    first = raw[0]

    # rank-first: 'Ah', 'TC', 'Kd', '9s'
    if first.upper() in RANK_CHARS:
        rank_ch = first.upper()
        suit_ch = SUIT_NORM.get(raw[-1])
        if suit_ch and rank_ch in RANK_VAL:
            return Card(RANK_VAL[rank_ch], suit_ch)

    # suit-first: 'SA', 'H2', 'DT', 'CK'
    if first.upper() in SUIT_CHARS:
        suit_ch = SUIT_NORM.get(first)
        rank_ch = raw[1:].upper()
        if suit_ch and rank_ch in RANK_VAL:
            return Card(RANK_VAL[rank_ch], suit_ch)

    raise ValueError(f"Cannot parse card: {card!r}")


def parse_all(cards) -> list:
    """Parse an iterable of cards into a list of Card objects."""
    return [parse(c) for c in cards]