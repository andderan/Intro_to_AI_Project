import gym
import random
import numpy as np

# Imports the Open AI gym enviroment
enviroment = "Blackjack-v0"
enviroment = gym.make(enviroment)

class Agent():
    def __init__(self, env, epsilon, gamma, alpha):
        self.env = enviroment
        self.qTable = dict()
        self.valid_actions = list(range(self.env.action_space.n))
        self.eps = epsilon
        self.discount = gamma
        self.learningRate = alpha

    def Init_Qtable_if_New(self, observation):
        if observation not in self.qTable:
            self.qTable[observation] = dict((action, 0.0) for action in self.valid_actions)

    def getMax(self, observation):
        self.Init_Qtable_if_New(observation)
        return max(self.qTable[observation].values())

    def GetAction(self, observation):
        self.Init_Qtable_if_New(observation)
        if random.random() > self.eps:
            maxQ = self.getMax(observation)
            ## Chooses a random action between all the best ones (incase of tie)
            action = random.choice([k for k in self.qTable[observation].keys()
                                    if self.qTable[observation][k] == maxQ])
        else:
            action = random.choice(self.valid_actions)

        self.eps = self.eps * .99
        return action

    def learn(self, observation, action, reward, next_observation):
        self.qTable[observation][action] += self.learningRate * (reward
                                                     + (self.discount * self.getMax(next_observation))
                                                     - self.qTable[observation][action])

### Model
agent = Agent(enviroment, 1, .01, .01)
total_games = 1000 #How many games per session
hands_per_game = 100 #How many hands are played in a game
avgPayout_per_hand = []

games = total_games
while games > 0:
    hand = hands_per_game
    game_payout = 0
    while hand > 0:
        state = enviroment.reset()
        done = False
        print("Game: {}, Hand: {}, Game Reward: {}, eps: {}".format(total_games- games + 1, hands_per_game - hand + 1,game_payout,agent.eps))
        while not done:
            print("STATE: ", state)
            action = agent.GetAction(state)
            nextState, reward, done, info = enviroment.step(action)
            agent.learn(state, action, reward, nextState)
            state = nextState
            game_payout += reward
            print("ACTION: ", action)
            print("NEWSTATE ", state)
        hand -= 1
        print("REWARD: ", reward)
    avgPayout_per_hand.append(game_payout)
    print("\n\n")
    games -= 1

##TODO Analysis
print ("Average payout after {} rounds is {}".format(100, sum(avgPayout_per_hand)/(1000)))
