# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random
import math

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util

class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """
    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        self.pacmanPosition = state.getPacmanPosition()
        self.ghostPositions = state.getGhostPositions()
        self.food = state.getFood()

        self.legalActions = state.getLegalPacmanActions()

    def __str__(self):
        # Create a string representation of the game state
        representation = f"Pac-Man Position: {self.pacmanPosition}\n"
        representation += f"Ghost Positions: {self.ghostPositions}\n"
        representation += "Food Locations:\n"
        # Assuming self.food is a grid or list of lists
        for row in self.food:
            representation += ''.join(['F' if cell else '.' for cell in row]) + '\n'
        return representation

class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.1,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.q_value = util.Counter() # Q values are stored in a dictionary
        self.lastState = [] # Most recent state stored
        self.lastAction = [] # Most recent action stored
        self.score = 0 # Current score stored

        # Exploration and exploitation
        self.explore = True
        
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.moves_made_total = 0
        self.moves_made_this_game = 0
        self.actionCounts = {}
        self.lastReward = None

        # Document training progress
        self.scores = []
        self.averageScores = []

        # Accessibility
        self.enable_print = False

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    def hashState(self, 
                  stateFeatures: GameStateFeatures) -> tuple:
            """
            Converts the game state features into a hashable format.

            Args:
                stateFeatures: An instance of GameStateFeatures, which encapsulates the current game state's significant features, such as Pacman's position, the positions of ghosts, and the food grid.

            Returns:
                A hashable representation of the state, incorporating Pacman's position, the ghost positions, and food locations.
            """

            # Ensure stateFeatures is of the correct type
            if not isinstance(stateFeatures, GameStateFeatures):
                raise TypeError("stateFeatures must be an instance of GameStateFeatures")

            # Convert state to hash
            foodGrid = stateFeatures.food
            foodList = foodGrid.asList()  # Assuming this exists and converts the grid to a list of coordinates
            stateHash = (stateFeatures.pacmanPosition, tuple(stateFeatures.ghostPositions), tuple(foodList))  # Convert list to tuple for hashability
            return stateHash

    def displayLeaderboard(self):
        """
        Prints the average scores over time, in segments of games, to provide a leaderboard-style overview of performance.

        This method assumes 'self.scores' contains a list of scores for each game played and 'self.enable_print' controls the printing.
        """
        if self.enable_print:
            print("Scores over time:")
        
        segment_size = 50
        # Calculate how many complete segments of 50 we have
        complete_segments = len(self.scores) // segment_size

        for segment in range(complete_segments):
            # Correctly calculate the start and end index for this segment
            start_index = segment * segment_size
            end_index = start_index + segment_size - 1  # Adjust for zero-based indexing
            # Calculate the average score for this segment
            avg_score = sum(self.scores[start_index:end_index + 1]) / segment_size

            if self.enable_print:
                print(f"Games {start_index + 1}-{end_index + 1}: Avg Score: {avg_score:.2f}")
        
        # Handle the last, potentially partial, segment
        remaining_scores = len(self.scores) % segment_size
        if remaining_scores > 0:
            start_index = complete_segments * segment_size
            avg_score = sum(self.scores[-remaining_scores:]) / remaining_scores

            if self.enable_print:
                print(f"Games {start_index + 1}-{len(self.scores)}: Avg Score: {avg_score:.2f}")

    def adjustEpsilonHalfway(self, 
                             currentEpisode, 
                             totalEpisodes):
        """
        Dynamically adjusts the exploration rate (epsilon) over the course of training.

        Args:
            currentEpisode: The current episode number during training.
            totalEpisodes: The total number of episodes allocated for training.

        This function linearly decreases epsilon during the first half of training and keeps it constant at its minimum value during the second half.
        """
        time_variable = 0.5
        max_exploration_rate = 0.5
        min_exploration_rate = 0.1

        if currentEpisode <= totalEpisodes * time_variable:
            # Linear decrease from 0.5 to 0.1 over the first half of training
            self.epsilon = max_exploration_rate - (currentEpisode / (totalEpisodes * time_variable)) * (max_exploration_rate - min_exploration_rate)
        else:
            # Keep epsilon constant at minimum value for the rest of training
            self.epsilon = 0.1
            

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        stateHash = self.hashState(state)
        if self.enable_print:
            print("Fetched Q-Value for", (stateHash, action), ":", self.q_value.get((stateHash, action), 0))
        return self.q_value.get((stateHash, action), 0)
        
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        legalActions = state.legalActions
        stateHash = self.hashState(state)
        q_values = [self.q_value.get((stateHash, action), 0) for action in legalActions]

        # Based on the list of legal Q-values, return the maximum value
        list_of_q_values = []
        for a in q_values:
            a = round(a, 2)
            list_of_q_values.append(a)
        return max(q_values) if q_values else 0

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures or None):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        stateHash = self.hashState(state)
        q = self.q_value.get((stateHash, action), 0)

        # If standard state, get the max Q-value for the next state
        if nextState is not None:
            maxQ = self.maxQValue(nextState)
        else:
            # If it's a terminal state, future rewards are 0. No action to take.
            if self.enable_print:
                print("Next state is terminal.")
            maxQ = 0

        # Visualise the Q-learning update
        if self.enable_print:
            print("Self.alpha:", self.alpha)
            print("Self.gamma:", self.gamma)
            print("Reward:", reward)
            print("Max Q-Value:", maxQ)
            print("Q-value before update for", (stateHash, action), ":", q.__round__(2))
            print("Q equation:", q.__round__(2), "+", self.alpha, "*", "(", reward, "+", self.gamma, "*", maxQ.__round__(2), "-", q.__round__(2), ")")
        
        # Update the Q-value for the state-action pair
        new_q = q + self.alpha * (reward + self.gamma * maxQ - q)
        self.q_value[(stateHash, action)] = new_q

        if self.enable_print:
            print("Q-value updated for", (stateHash, action), ":", new_q.__round__(2))

    # Count-based Exploration function

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        stateHash = self.hashState(state)

        key = (stateHash, action)
        if key not in self.actionCounts:
            self.actionCounts[key] = 0
        self.actionCounts[key] += 1

        if self.enable_print:
            print("Action Counts updated from:", self.actionCounts.get(key, 0) - 1, "to", self.actionCounts.get(key, 0))
        
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        stateHash = self.hashState(state)

        key = (stateHash, action)

        # if self.enable_print:
        # print("Action Counts fetched:", self.actionCounts.get(key, 0))

        return self.actionCounts.get(key, 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        exploration_bonus = 1 / (counts + 1)
        adjusted_utility = utility + (exploration_bonus*2)
        return adjusted_utility

    def getClosestGhostDistance(self, state: GameState) -> float:
        pacmanPosition = state.getPacmanPosition()
        ghostDistances = [
            math.sqrt((pacmanPosition[0] - ghostPos[0]) ** 2 + (pacmanPosition[1] - ghostPos[1]) ** 2)
            for ghostPos in state.getGhostPositions()
        ]
        return min(ghostDistances)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # We get the features of the state like this
        stateFeatures = GameStateFeatures(state)
        # print("State Features:", stateFeatures)

        # The data we have about the state of the game
        legalActions = stateFeatures.legalActions
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        if self.enable_print:
            print("|\n")
            print(f"Game number: {self.getEpisodesSoFar()}, Moves this game: {self.moves_made_this_game}, Total moves: {self.moves_made_total}")

        # Print the current reward based on the score difference.
        reward = state.getScore() - self.score
        if self.enable_print:
            print("Reward for last turn: ", reward) # Reward is the difference in score between the current state and the last state

        # If this is not the first action of the game, learn from the last action
        if len(self.lastState) > 0:
            last_state = self.lastState[-1]
            last_action = self.lastAction[-1]

            # Learn from the action
            self.learn(last_state, last_action, reward, stateFeatures)

        self.moves_made_this_game += 1
        self.moves_made_total += 1

        if self.enable_print:
            print("\n")
            print("Now we can focus on this turn\n")
            # Visualise the state of the game
            print("New State: ")
            print(state)
            print("Legal moves:", legalActions)
        
        # Decide on the next action to take.
                    
        # Epsilon-greedy implementation also using explorationFn(utility, counts)
        if util.flipCoin(self.epsilon):
            # Exploration I (with probability epislon = 0.5): Choose a random action
            action = random.choice(legalActions)

            if self.enable_print:
                print(f"Coinflip == Exploration: Choosing a random action: {action}")

        else:
            # Exploration II: use count-utility bonus to select the action
            if self.explore:
                adjusted_q_values = {}
                for action in legalActions:
                    q_value = self.getQValue(stateFeatures, action)
                    count = self.getCount(stateFeatures, action)
                    adjusted_q_values[action] = self.explorationFn(q_value, count)

                    if self.enable_print:
                        print(f"! Q-Value for action {action}: {q_value}, Count: {count}")

                # Select the action with the highest adjusted Q-value
                max_adjusted_q_value = max(adjusted_q_values.values())
                best_actions = [action for action, q in adjusted_q_values.items() if q == max_adjusted_q_value]
                action = random.choice(best_actions)
                if self.enable_print:
                    print("! Adjusted Q-Values:", adjusted_q_values)
                    print(f"! Chosen action: {action}")
            else:
                # Pure exploitation for inference at runtime (no exploration)
                q_values = {}
                for action in legalActions:
                    q_value = self.getQValue(stateFeatures, action)
                    print(f"Pure exploitation: Q-Value for action {action}: {q_value}")
                    q_values[action] = q_value
                
                # Select the action with the highest Q-value
                max_q_value = max(q_values.values())
                # Robust to multiple actions with the same max Q-value
                actionsWithMaxQ = [action for action, q in q_values.items() if q == max_q_value]
                action = random.choice(actionsWithMaxQ)
                print(f"Pure exploitation: Chosen action: {action}")
                
        self.updateCount(stateFeatures, action)  # Update the count for the chosen action
        self.lastState.append(stateFeatures)
        self.lastAction.append(action)
        self.score = state.getScore()

        # The current code shows how to do that but just makes the choice randomly.
        if self.enable_print:
            print(f"Updated count for action {action}: {self.getCount(stateFeatures, action)}")            
            print("")
        
        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        if self.enable_print:
            print("\n----------------------------------------\n")
            print(f"Game {self.getEpisodesSoFar()} just ended!")

        if not self.enable_print:
            print(f"Game {self.getEpisodesSoFar()} ✔ {state.getScore()}")

        # Calculate reward for the last action taken
        lastScore = self.score
        lastState = self.lastState[-1]
        lastAction = self.lastAction[-1]

        currentScore = state.getScore()
        reward = currentScore - lastScore
        self.scores.append(currentScore)

        # Call learn with a dummy 'nextState' since there isn't one
        self.learn(lastState, lastAction, reward, None)

        self.displayLeaderboard()

        if self.enable_print:
            print("\n----------------------------------------\n")
            print(f"Starting game {self.getEpisodesSoFar() + 1}!\n")
            print("episodesSoFar:", self.episodesSoFar)
            print("New epsilon:", self.epsilon, "\n")

        # Reset attributes for the next episode
        self.score = 0
        self.moves_made_this_game = 0
        self.lastState = []  # Reset to None or a new GameStateFeatures object if needed
        self.lastAction = []

        # Track the number of episodes and adjust learning parameters if training is done
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() >= self.getNumTraining():
            msg = 'Training Done (turning off epsilon, alpha, and explore fnct)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)  # Consider whether you really want to set alpha to 0 or just decrease it
            self.setEpsilon(0)
            self.enable_print = True
            self.explore = False
        else:
            # Only adjust epsilon if training is not yet complete
            self.adjustEpsilonHalfway(self.getEpisodesSoFar(), self.getNumTraining())
            self.explore = True

