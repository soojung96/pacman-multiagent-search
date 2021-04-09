# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print("successor GameState: ", str(successorGameState))
        # print("new position: ", str(newPos))
        # print("new food: ", str(newFood))
        # print("new ghost states: ", str(newGhostStates))
        # print("new scared times: ", str(newScaredTimes))

        #if there are less foods than before, get one good point
        #if you are closer to a food, get one good point
        #distance to closest ghost
        #score of successor state
        listOfFood = newFood.asList()
        distanceToFood = 0
        if len(listOfFood) != 0:
            distanceToFood = 1000000
            for foodPos in listOfFood:
                foodDist = manhattanDistance(newPos, foodPos)
                if foodDist < distanceToFood:
                    distanceToFood = foodDist

        currentFoodNum = currentGameState.getNumFood()
        successorFoodNum = successorGameState.getNumFood()

        eatFoodPellet = currentFoodNum - successorFoodNum

        # distanceToGhost = 0
        # dist = 1000000
        # for ghostState in newGhostStates:
        #     if ghostState.scaredTimer == 0:
        #         ghostPos = ghostState.getPosition()
        #         ghostDist = manhattanDistance(newPos, ghostPos)
        #         dist = min(ghostDist, dist)
        # if len(newGhostStates) > 0:
        #     distanceToGhost = dist

        successorScore = successorGameState.getScore()
        return successorScore - distanceToFood + 10 * eatFoodPellet


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.max(gameState, self.depth, 1)

    def max(self, gameState, depth, ghostIndex):
        if depth <= 0:
            return self.evaluationFunction(gameState)
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        pacmanActions = gameState.getLegalActions(0)
        bestScore = float('-inf')
        bestAction = Directions.STOP

        for action in pacmanActions:
            successorState = gameState.generateSuccessor(0, action)
            score = self.min(successorState, depth, ghostIndex)
            if score > bestScore:
                bestAction = action
                bestScore = score
        if depth == self.depth:
            return bestAction
        else:
            return bestScore

    def min(self, gameState, depth, ghostIndex):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        numGhosts = gameState.getNumAgents() - 1
        bestScore = float('inf')
        bestAction = Directions.STOP

        ghostActions = gameState.getLegalActions(ghostIndex)
        for action in ghostActions:
            successorState = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex < numGhosts:
                score = self.min(successorState, depth, ghostIndex + 1)
            elif ghostIndex == numGhosts:
                score = self.max(successorState, depth - 1, 1)
            if score < bestScore:
                bestAction = action
                bestScore = score
        return bestScore



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.max(gameState, self.depth, 1, float("-inf"), float("inf"))

    def max(self, gameState, depth, ghostIndex, alpha, beta):
        if depth <= 0:
            return self.evaluationFunction(gameState)
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        pacmanActions = gameState.getLegalActions(0)
        bestScore = float('-inf')
        bestAction = Directions.STOP

        for action in pacmanActions:
            successorState = gameState.generateSuccessor(0, action)
            score = self.min(successorState, depth, ghostIndex, alpha, beta)
            if score > bestScore:
                bestAction = action
                bestScore = score
            if bestScore > beta:
                if depth == self.depth:
                    return bestAction
                else:
                    return bestScore
            alpha = max(alpha, bestScore)
        if depth == self.depth:
            return bestAction
        else:
            return bestScore

    def min(self, gameState, depth, ghostIndex, alpha, beta):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        numGhosts = gameState.getNumAgents() - 1
        bestScore = float('inf')
        bestAction = Directions.STOP

        ghostActions = gameState.getLegalActions(ghostIndex)
        for action in ghostActions:
            successorState = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex < numGhosts:
                score = self.min(successorState, depth, ghostIndex + 1, alpha, beta)
            elif ghostIndex == numGhosts:
                score = self.max(successorState, depth - 1, 1, alpha, beta)
            if score < bestScore:
                bestAction = action
                bestScore = score
            if bestScore < alpha:
                return bestScore
            beta = min(beta, bestScore)
        return bestScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.max(gameState, self.depth, 1)

    def max(self, gameState, depth, ghostIndex):
        if depth <= 0:
            return self.evaluationFunction(gameState)
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        pacmanActions = gameState.getLegalActions(0)
        bestScore = float('-inf')
        bestAction = Directions.STOP

        for action in pacmanActions:
            successorState = gameState.generateSuccessor(0, action)
            score = self.min(successorState, depth, ghostIndex)
            if score > bestScore:
                bestAction = action
                bestScore = score
        if depth == self.depth:
            return bestAction
        else:
            return bestScore

    def min(self, gameState, depth, ghostIndex):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        numGhosts = gameState.getNumAgents() - 1
        bestScore = float('inf')
        bestAction = Directions.STOP

        ghostActions = gameState.getLegalActions(ghostIndex)
        ghostScores = []
        for action in ghostActions:
            successorState = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex < numGhosts:
                score = self.min(successorState, depth, ghostIndex + 1)
            elif ghostIndex == numGhosts:
                score = self.max(successorState, depth - 1, 1)
            ghostScores.append(score)
        avgScore = sum(ghostScores) / len(ghostScores)
        return avgScore


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    #if there are less foods than before, get one good point
    #if you are closer to a food, get one good point
    #distance to closest ghost
    #score of successor state
    listOfFood = food.asList()
    distanceToFood = 0
    if len(listOfFood) != 0:
        distanceToFood = 1000000
        for foodPos in listOfFood:
            foodDist = manhattanDistance(pos, foodPos)
            if foodDist < distanceToFood:
                distanceToFood = foodDist

    currentFoodNum = currentGameState.getNumFood()

    # distanceToGhost = 0
    # dist = 1000000
    # for ghostState in newGhostStates:
    #     if ghostState.scaredTimer == 0:
    #         ghostPos = ghostState.getPosition()
    #         ghostDist = manhattanDistance(newPos, ghostPos)
    #         dist = min(ghostDist, dist)
    # if len(newGhostStates) > 0:
    #     distanceToGhost = dist

    score = currentGameState.getScore()
    return score - distanceToFood - currentFoodNum


# Abbreviation
better = betterEvaluationFunction
