import random
 
from bke import MLAgent, is_winner, opponent, RandomAgent, train_and_plot
 
 
class MyAgent(MLAgent):
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1
        elif is_winner(board, opponent[self.symbol]):
            reward = -1
        else:
            reward = 0
        return reward
    
    
random.seed(1)
 
my_agent = MyAgent(alpha=0.8, epsilon=0.2)
random_agent = RandomAgent()
 
train_and_plot(
    agent=my_agent,
    validation_agent=random_agent,
    iterations=50,
    trainings=100,
    validations=1000)

# Alpha geeft aan hoe snel de agent oude informatie weggooit voor nieuwe om zo te leren
# Epsilon geeft aan hoe veel van de oude informatie de agent gebruikt bij het proberen.
# Met alpha(0.8) en epsilon(0.2) is de agent na ongeveer 2800 spellen uitgetrained
# Als je alpha en epsilon omdraaid is de agent uitgeleerd na ongeveer 2300 spellen en geeft ook een hogere winrate
