class HumanPlayer:
    """ Human Poker Player"""

    # Player constructor
    def __init__(self):
        self.bet = 0
        self.raised = False
        self.reset()

    def reset(self):
        self.bet =0
        self.raised = False

    # Parameters private_card, legal_actions, play_first are printed in the environment and are not needed here
    def action_1(self):
        action = input("Choose player action:").strip().lower()
        return action