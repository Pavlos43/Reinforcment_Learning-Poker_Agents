import numpy as np
import random as rnd

class FixedOpponent:
    """
    Fixed AI opponent styles. Namely:
        - PAL_BOT: The game strategy of our dear friend Mitsos.
        - RANDOM Player: Chooses uniformly at random among legal actions.
        - TIGHT Player: Bluffs rarely. Plays kind of strongly only with good private cards or good public-card matches. Folds more often with weak hands.
        - LIMPING player: Checks or calls whenever possible. Folds with small probability.
        - AGGRESSIVE/LOOSE Player: Raises whenever possible. Otherwise calls/checks.
        - HONEST-STRENGTH Player: Bluffs 5% of the time. Is aggressive depending on the quality of the cards.
    """

    # Player constructor
    def __init__(self, style):
        self.bet=0
        self.raised=False
        self.style = style      # style = {"Pal-Bot", "Random", "Tight", "Limping", "Aggressive", "Honest-strength"}
        self.reset()

    def reset(self):
        self.bet = 0
        self.raised = False

    # Action a player can take during the first betting round => "Check/Call", "Raise", "Fold" allowed
    def action_1(self, private_card, legal_actions, play_first):

        # ============================================== #
        # When playing first => "Raise" if legal         #
        # When private card is 'K' => "Raise" if legal   #
        # Otherwise, "Call"/"Check"                      #
        # ============================================== #
        if self.style == "Pal-Bot":
            if (private_card == 'K' or play_first == 0) and "raise" in legal_actions:
                action = "raise"
            elif "call" in legal_actions:
                action = "call"
            else:
                action = "check"


        # =================================================================================================================== #
        # Choose uniformly among actions "Check/Call", "Raise", "Fold"                                                        #
        # If no raises have occurred, calling covers the check decision by adding 0 chips to the pot                          #
        # =================================================================================================================== #
        elif self.style == "Random":
            action = rnd.choice(legal_actions)


        # ====================================================================================================================== #
        # If card is 'J': Choose "Check" if legal. If not, "Fold" if legal. If not and facing a bet, "Call".                     #
        # If card is 'Q': Choose "Check" if legal. If facing a bet, choose "Call" if legal. Otherwise, "Fold".                   #
        # If card is 'K': Choose "Raise" with prob = 0.6 if legal. Otherwise, if facing a bet, "Call" if legal. If not, "Check". #
        # ====================================================================================================================== #
        elif self.style == "Tight":

            x = np.random.rand()

            if x< 0.03 and "raise" in legal_actions:
                action = "raise"
            elif private_card == 'J':
                if "check" in legal_actions:
                    action = "check"
                elif "fold" in legal_actions:
                    action = "fold"
                else:
                    action = "call"
                    
            elif private_card == 'Q':
                if "check" in legal_actions:
                    action = "check"
                elif "call" in legal_actions:
                    action = "call"
                else:
                    action = "fold"
            
            # private_card = 'K'
            else:
                if ("raise" in legal_actions and x < 0.6):
                    action = "raise"
                elif "call" in legal_actions:
                    action = "call"
                else:
                    action = "check"


        # ========================================================================= #
        # If card is 'K', choose "Raise" with prob = 0.4 if legal.                  #
        # Regardless of the card, choose "Raise" with prob = 0.02 if legal.         #
        # Otherwise, if facing a bet, "Call". If not (and check is legal), "Check". #
        # ========================================================================= #
        elif self.style == "Limping":

            x = np.random.rand()
            if (private_card == 'K' and x < 0.4 and "raise" in legal_actions):
                action = "raise"
            elif (x < 0.02 and "raise" in legal_actions):
                action = "raise"
            elif "call" in legal_actions:
                action = "call"
            else:
                action = "check"


        # ========================================================================= #
        # Choose "Raise: with prob = 0.65 if legal.                                 #
        # If not, choose "Call" with prob = 0.85 if legal/facing a bet.             #
        # If not, choose "Check" if legal. Otherwise, choose "Fold"                 #
        # ========================================================================= #
        elif self.style == "Aggressive":
            x = np.random.rand()
            if (x < 0.65 and "raise" in legal_actions):
                action = "raise"
            elif (x < 0.85 and "call" in legal_actions):
                action = "call"
            elif "check" in legal_actions:
                action = "check"
            else:
                action = "fold"


        # self.style = "Honest-strength"
        # ================================================================================================================================================ #
        # If card is 'J': "Raise" with prob = 0.05 (bluff) if legal. If not, "Check" if legal. If not, "Fold" if legal. Otherwise, "Call".                 #
        # If card is 'Q': "Raise" with prob = 0.05 (bluff) if legal. If facing a bet, "Call" with prob = 0.6 if legal. If not, "Fold". Otherwise, "Check". #
        # If card is 'K': "Raise" with prob = 0.5 if legal. If not, choose "Call" (if facing a bet) or "Check".                                            #
        # ================================================================================================================================================ #
        else:
            x = np.random.rand()
            if private_card == 'J':
                if (x < 0.05 and "raise" in legal_actions):
                    action = "raise"
                elif "check" in legal_actions:
                    action = "check"
                elif "fold" in legal_actions:
                    action = "fold"
                else:
                    action = "call"
            elif private_card == 'Q':
                if (x < 0.05 and "raise" in legal_actions):
                    action = "raise"
                elif (x < 0.65 and "call" in legal_actions):
                    action = "call"
                elif "fold" in legal_actions:
                    action = "fold"
                elif "call" not in legal_actions and "check" in legal_actions:
                    action = "check"
                else:
                    action ="call"
            # private_card = 'K'
            else:
                if ("raise" in legal_actions and x < 0.5):
                    action = "raise"
                elif "call" in legal_actions:
                    action = "call"
                else:
                    action = "check"

        return action

    # Action a player can take during the second betting round => "Check/Call", "Raise", "Fold" allowed
    def action_2(self, private_card, legal_actions, play_first, table_card):

        # ====================================================================================================================================== #
        # When playing first => "Raise" if legal and private card is 'K'. If not, "Check"/"Call".                                                #
        # When pair 'QQ','KK' => "Raise" if legal. If not, "Check/Call".                                                                         #
        # When pair 'JJ' => "Raise" with probability 3/5, otherwise "Check"/"Call" (bluff bait).                                                 #
        # If card is 'J' and there's no pair, "Check" if legal. If facing a bet, "Fold/Raise" with prob = 0.5 (bluff yourself or accept defeat). #
        # Otherwise, "Call"/"Check".                                                                                                             #
        # ====================================================================================================================================== #
        if self.style == "Pal-Bot":
            if (private_card == 'K' and play_first == 0):
                if "raise" in legal_actions:
                    action = "raise"
                elif ("check" in legal_actions):
                    action = "check"
                else:
                    action = "call"
            elif (private_card == 'J' and table_card == 'J'):
                x = np.random.rand()
                if (x < 0.6 and "raise" in legal_actions):
                    action = "raise"
                else:
                    action = "check" if ("check" in legal_actions) else "call"

            # Pair but not 'JJ' => 'QQ' or 'KK'
            elif private_card == table_card:
                if "raise" in legal_actions:
                    action = "raise"
                else:
                    action = "check" if ("check" in legal_actions) else "call"                    
            
            # 'J' but no pair => worst possible outcome
            elif (private_card == 'J' and table_card != 'J'):
                if "check" in legal_actions:
                    action = "check"
                else:
                    x = np.random.rand()
                    action = "raise" if (x < 0.5 and "raise" in legal_actions) else "fold"
            
            # No pair, no 'K', no 'J' => 'Q'
            else:
                action = "call" if ("call" in legal_actions) else "check"


        # =================================================================================================================== #
        # Choose uniformly among actions "Check/Call", "Raise", "Fold".                                                       #
        # If you don't have enough chips, fold or go all-in and you get the equivalent share of the bid if you win.           #
        # If no raises have occurred, calling covers the check decision by adding 0 chips to the pot.                         #
        # =================================================================================================================== #
        elif self.style == "Random":
            action = rnd.choice(legal_actions)


        # =================================================================================================================== #
        # "Raise" with prob = 0.03 if legal. (bluff)                                                                          #
        # When pair of any rank => "Raise" if legal. If not, "Call"/"Check".                                                  #
        # If card is 'J' and there's no pair, "Check" if legal. If not, "Fold" if legal. Otherwise, "Call".                   #
        # If card is 'Q' and there's no pair, "Check" if legal. If facing a bet, "Call". Otherwise, "Fold" if legal.          #
        # If card is 'K' and there's no pair, "Raise" with prob = 0.6 if legal. If not, "Call"/"Check".                       #
        # =================================================================================================================== #
        elif self.style == "Tight":
            x = np.random.rand()
            
            if (x < 0.03 and "raise" in legal_actions):
                action = "raise"
            elif private_card == table_card:
                if "raise" in legal_actions:
                    action = "raise"
                else:
                    action = "call" if ("call" in legal_actions) else "check"
            
            elif private_card == 'J':
                if "check" in legal_actions:
                    action = "check"
                else:
                    action = "fold" if ("fold" in legal_actions) else "call"
            
            elif private_card == 'Q':
                if "check" in legal_actions:
                    action = "check"
                else:
                    action = "call" if ("call" in legal_actions) else "fold"
            
            # private_card == 'K'
            else:
                if (x < 0.6 and "raise" in legal_actions):
                    action = "raise"
                else:
                    action = "call" if ("call" in legal_actions) else "check"


        # ============================================================================== #
        # Regardless of cards, "Raise" with prob = 0.02 (bluff).                         #
        # When pair of any rank or private card = 'K', "Raise" with prob = 0.4 if legal. #
        # If facing a bet, "Call". If not, "Check".                                      #
        # ============================================================================== #
        elif self.style == "Limping":
            x = np.random.rand()
            if ((x < 0.4) and ("raise" in legal_actions) and (private_card == 'K' or private_card == table_card)):
                action = "raise"
            elif (x< 0.02 and "raise" in legal_actions):
                    action = "raise"
            elif "call" in legal_actions:
                action = "call"
            else:
                action = "check"

        
        # ========================================================================= #
        # Choose "Raise: with prob = 0.65 if legal.                                 #
        # If not, choose "Call" with prob = 0.85 if legal/facing a bet.             #
        # If not, choose "Check" if legal. Otherwise, choose "Fold"                 #
        # ========================================================================= #
        elif self.style == "Aggressive":
            x = np.random.rand()
            if (x < 0.65 and "raise" in legal_actions):
                action = "raise"
            elif (x < 0.85 and "call" in legal_actions):
                action = "call"
            elif "check" in legal_actions:
                action = "check"
            else:
                action = "fold"


        # self.style = "Honest-strength"
        # ================================================================================================================================================ #
        # When pair of any rank, "Raise" with prob = 0.9 if legal. If not, "Call"/"Check".                                                                 #
        # If card is 'J': "Raise" with prob = 0.05 (bluff) if legal. If not, "Check" if legal. If not, "Fold" if legal. Otherwise, "Call".                 #
        # If card is 'Q': "Raise" with prob = 0.05 (bluff) if legal. If facing a bet, "Call" with prob = 0.6 if legal. If not, "Fold". Otherwise, "Check". #
        # If card is 'K': "Raise" with prob = 0.5 if legal. If not, choose "Call" (if facing a bet) or "Check".                                            #
        # ================================================================================================================================================ #
        else:
            x = np.random.rand()
            if private_card == table_card:
                if (x < 0.9 and "raise" in legal_actions):
                    action = "raise"
                elif "call" in legal_actions:
                    action = "call"
                else:
                    action = "check"

            elif private_card == 'J':
                if (x < 0.05 and "raise" in legal_actions):
                    action = "raise"
                elif "check" in legal_actions:
                    action = "check"
                elif "fold" in legal_actions:
                    action = "fold"
                else:
                    action = "call"

            elif private_card == 'Q':
                if (x < 0.05 and "raise" in legal_actions):
                    action = "raise"
                elif (x < 0.65 and "call" in legal_actions):
                    action = "call"
                elif "fold" in legal_actions:
                    action = "fold"
                elif "call" not in legal_actions and "check" in legal_actions:
                    action = "check"
                else:
                    action ="call"
            
            # private_card = 'K'
            else:
                if ("raise" in legal_actions and x < 0.5):
                    action = "raise"
                elif "call" in legal_actions:
                    action = "call"
                else:
                    action = "check"

        return action