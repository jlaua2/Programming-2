#Please do not add any imports
#This particularly includes numpy!
import math

############################## PROBLEM 1 ######################################

#This function should take a matrix with your payoffs for the game and your
#opponent's current strategy and compute your expected utility for each action.
#This will be useful in implementing the learning dynamics
def expectedValues(game,opponentStrategy):
    num_actions = len(game)
    num_opp_actions = len(opponentStrategy)
    evs = []
    for i in range(num_actions):
        ev = sum(game[i][j] * opponentStrategy[j] for j in range(num_opp_actions))
        evs.append(ev)
    return evs


############################## PROBLEM 2 ######################################

#This function should implement one iteration of best response dynamics for
#a player.  It takes that players's payoff matrix and the opponent's strategy
#and returns a best response
def bestResponseDynamics(game,opponentStrategy):
    evs = expectedValues(game, opponentStrategy)
    max_ev = max(evs)
    # Strategy: 1.0 for the first action that achieves the max expected value
    strategy = [0.0] * len(game)
    for i in range(len(evs)):
        if evs[i] == max_ev:
            strategy[i] = 1.0
            break
    return strategy

############################## PROBLEM 3 ######################################

#This class should implement fictitious play for one player.
class FictitiousPlay:
    #You shouldn't need to change __init__
    def __init__(self,game):
        self.history = [0.0 for i in range(len(game[0]))]
        self.game = game
        self.iterations = 0

    #This should perform one iteration of fictitious play
    def updateStrategy(self, opponentStrategy):
        # Update history
        for i in range(len(opponentStrategy)):
            self.history[i] += opponentStrategy[i]
        self.iterations += 1
        
        # Empirical distribution of opponent
        empirical_dist = [x / self.iterations for x in self.history]
        
        # Best response to that distribution
        return bestResponseDynamics(self.game, empirical_dist)

############################## PROBLEM 4 ######################################

#This class should implement smoothed fictitious play for one player.
class SmoothedFictitiousPlay:
    #You shouldn't need to change __init__
    def __init__(self,game,gamma):
        self.history = [0.0 for i in range(len(game[0]))]
        self.game = game
        self.gamma = gamma
        self.iterations = 0

    #This should perform one iteration of smoothed fictitious play
    def updateStrategy(self, opponentStrategy):
        for i in range(len(opponentStrategy)):
            self.history[i] += opponentStrategy[i]
        self.iterations += 1
        
        empirical_dist = [x / self.iterations for x in self.history]
        evs = expectedValues(self.game, empirical_dist)
        
        # Logit response (Softmax)
        # Strategy_i = exp(ev_i / gamma) / sum(exp(ev_j / gamma))
        shifted_evs = [v / self.gamma for v in evs]
        # Subtract max for numerical stability
        max_v = max(shifted_evs)
        exps = [math.exp(v - max_v) for v in shifted_evs]
        sum_exps = sum(exps)
        
        return [e / sum_exps for e in exps]

############################## PROBLEM 5 ######################################

#This class should implement regret matching for one player.
class RegretMatching:
    #You shouldn't need to change __init__
    def __init__(self,game):
        self.regretSums = [0.0 for i in range(len(game))]
        self.game = game
        self.last_strategy = [1.0/len(game)] * len(game)

    def regretSumsToStrategy(self):
        positive_regrets = [max(0.0, r) for r in self.regretSums]
        total_regret = sum(positive_regrets)
        if total_regret > 0:
            return [r / total_regret for r in positive_regrets]
        else:
            return [1.0 / len(self.game)] * len(self.game)

    def updateStrategy(self, opponentStrategy):
        # Calculate utility of every action
        action_utilities = expectedValues(self.game, opponentStrategy)
        # Utility of the strategy we actually played last time
        actual_utility = sum(self.last_strategy[i] * action_utilities[i] for i in range(len(self.game)))
        
        # Accumulate regret
        for i in range(len(self.game)):
            self.regretSums[i] += (action_utilities[i] - actual_utility)
            
        self.last_strategy = self.regretSumsToStrategy()
        return self.last_strategy

############################## PROBLEM 6 ######################################

# Prior that causes Matching Pennies to cycle (e.g. slight bias)
MPPrior1 = [1.0, 0.0]
MPPrior2 = [0.0, 1.0]

############################## PROBLEM 7 ######################################

# Prior for Shapley's game to stay in a cycle
ShapleyPrior1 = [1.0, 0.0, 0.0]
ShapleyPrior2 = [0.0, 1.0, 0.0]

############################## PROBLEM 8 ######################################

# Coordination game with a bias
P8Game1 = [[2, 0], [0, 1]]
P8Game2 = [[2, 0], [0, 1]]
P8Prior1 = [0.4, 0.6]
P8Prior2 = [0.6, 0.4]

############################## PROBLEM 9 ######################################

# Game where BR self-play hits (0,0) immediately but FP drifts
P9Game1 = [[1, 0], [0, 0]]
P9Game2 = [[1, 0], [0, 0]]
P9Prior1 = [0.0, 1.0]
P9Prior2 = [0.0, 1.0]


############################## PROBLEM 10 ######################################

# Game where Smoothed FP converges to Mixed NE but Regret Matching oscillates
P10Game1 = [[0, 1], [1, 0]]
P10Game2 = [[1, 0], [0, 1]]
P10Prior1 = [0.5, 0.5]
P10Prior2 = [0.5, 0.5]

############################## PROBLEM 11 ######################################

#This class should implement optimistic regret matching for one player.
class OptimisticRegretMatching:
    #You shouldn't need to change __init__
    def __init__(self,game):
        self.regretSums = [0.0 for i in range(len(game))]
        self.lastRegrets = [0.0 for i in range(len(game))]
        self.game = game
        self.last_strategy = [1.0/len(game)] * len(game)

    def regretSumsToStrategy(self):
        # Optimistic: use current regrets + last regrets as a proxy for "next"
        opt_regrets = [max(0.0, self.regretSums[i] + self.lastRegrets[i]) for i in range(len(self.game))]
        total = sum(opt_regrets)
        if total > 0:
            return [r / total for r in opt_regrets]
        else:
            return [1.0 / len(self.game)] * len(self.game)

    def updateStrategy(self, opponentStrategy):
        action_utilities = expectedValues(self.game, opponentStrategy)
        actual_utility = sum(self.last_strategy[i] * action_utilities[i] for i in range(len(self.game)))
        
        current_regrets = []
        for i in range(len(self.game)):
            regret = action_utilities[i] - actual_utility
            current_regrets.append(regret)
            self.regretSums[i] += regret
            
        self.lastRegrets = current_regrets
        self.last_strategy = self.regretSumsToStrategy()
        return self.last_strategy
    