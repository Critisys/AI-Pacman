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
        pac_pos = successorGameState.getPacmanPosition()
        food_pos = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        food_dis_lst = [manhattanDistance(pac_pos, food) for food in food_pos.asList()]
        evaluate =-len(food_pos.asList())*100
        if len(food_pos.asList()) != 0:
            evaluate -= min(food_dis_lst)
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0:
                distance = manhattanDistance(pac_pos, ghost.getPosition())
                if distance <= 1:
                    evaluate = -1e9
        return evaluate

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
        
        numAgent = gameState.getNumAgents()
        evaluate = []
                
        def miniMax(gState, counter, eval_lst):
            if counter >= self.depth * numAgent or gState.isLose() or  gState.isWin() :
                return self.evaluationFunction(gState)
            if counter % numAgent == 0 : 
                max_value = -1e9
                for child in gState.getLegalActions(counter % numAgent):
                    max_value = max(max_value, miniMax(gState.generateSuccessor(counter % numAgent, child), counter + 1,eval_lst))
                    if counter == 0:
                        eval_lst.append(max_value)
                return max_value
            else: 
                min_value = 1e9
                for child in gState.getLegalActions(counter % numAgent):
                    min_value = min(min_value, miniMax(gState.generateSuccessor(counter % numAgent, child), counter + 1,eval_lst))
                return min_value
        
        x = miniMax(gameState,0,evaluate)
        return gameState.getLegalActions(0)[evaluate.index(x)]
        
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgent = gameState.getNumAgents()
        evaluate = []
        
        def alphabeta(gState, alpha, beta,counter,eval_lst):
            if counter >= self.depth * numAgent or gState.isLose() or  gState.isWin() :
                return self.evaluationFunction(gState)
            if counter % numAgent == 0 : #max
                max_value = -1e9
                for child in gState.getLegalActions(counter % numAgent):
                    max_value = max(max_value, alphabeta(gState.generateSuccessor(counter % numAgent, child), alpha, beta, counter + 1,eval_lst))
                    alpha = max(alpha, alphabeta(gState.generateSuccessor(counter % numAgent, child),alpha, beta, counter + 1,eval_lst))
                    if beta < alpha:
                        break
                    if counter == 0 :
                        eval_lst.append(max_value)
                return max_value
            else: #min
                min_value = 1e9
                for child in gState.getLegalActions(counter % numAgent):
                    min_value = min(min_value, alphabeta(gState.generateSuccessor(counter % numAgent, child), alpha, beta, counter + 1,eval_lst))
                    beta = min(beta, alphabeta(gState.generateSuccessor(counter % numAgent, child), alpha, beta, counter + 1,eval_lst))
                    if beta < alpha:
                        break
                return min_value
        
        x = alphabeta(gameState, -1e9, 1e9 ,0,evaluate)
        return gameState.getLegalActions(0)[evaluate.index(x)]
        util.raiseNotDefined()

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
        numAgent = gameState.getNumAgents()
        evaluate = []

        def expectimax(gState, counter, eval_lst):
            if counter >= self.depth * numAgent or gState.isLose() or gState.isWin():
                return self.evaluationFunction(gState)
            if counter % numAgent == 0 : 
                max_value = -1e9
                for child in gState.getLegalActions(counter % numAgent):
                    max_value = max(max_value, expectimax(gState.generateSuccessor(counter % numAgent, child), counter + 1,eval_lst))
                    if counter == 0:
                        eval_lst.append(max_value)
                return max_value
            else:
                Average = 0
                for child in gState.getLegalActions(counter % numAgent):
                    Average = Average + expectimax(gState.generateSuccessor(counter % numAgent, child), counter + 1,eval_lst)
                return Average/len(gState.getLegalActions(counter % numAgent))
        
        x = expectimax(gameState,0,evaluate)
        return gameState.getLegalActions(0)[evaluate.index(x)]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <
        Our evaluation will be the sum of:  current_score + food_evaluation + ghost_evaluation + capsules_evaluation
                                            - current_number_of_food*20 - current_number_of_capsules*1000

        The reason for number of capsuples having a high factor of 1000 is because we want pacman to prioritize 
        in consuming the capsules when pacman is right next to it.
        
        Same reason goes for the factor of food number.

        food evaluation:    The nearest food to pacman will give the highest value.

        capsules evaluation:    The nears capsules to pacman will give the highest value. Capsules also have 
                                a higher factor than food.

        ghost evaluation:   If ghost is 2 unit away from pacman and not scared, this evaluation will be 10^10 negative
                            so that every action pacman took that take him far away from ghost will always return a betters
                            evaluation. if ghost is more than 2 unit away this evaluation will be zero.
                            
                            If ghost is scared however. 16 unit is the maximun distance from pacman to ghost that
                            give a positive evaluation so pacman will have a tendency to comsume ghost and gain more points.
                            Farther than that this will return a zero value there for pacman will focus more on food.
    >
    """
    PacPos = currentGameState.getPacmanPosition()
    ghost_eval = 0
    food_eval = 0
    capsules_eval = 0
    
    food_dis = [-manhattanDistance(PacPos, food) for food in currentGameState.getFood().asList()]
    if len(food_dis) != 0:
        food_eval += max(food_dis)
        
    capsules_dis = [-5*manhattanDistance(PacPos, capsules) for capsules in currentGameState.getCapsules()]
    if len(capsules_dis) != 0:
        capsules_eval += max(capsules_dis)
    
    for ghostState in currentGameState.getGhostStates():
            distance = manhattanDistance(PacPos, ghostState.getPosition())
            if ghostState.scaredTimer == 0:
                if distance <= 2:
                    ghost_eval = -1e10
            else:
                ghost_eval = 10*(max(16 - distance,0))

    return currentGameState.getScore() + food_eval + capsules_eval + ghost_eval - currentGameState.getNumFood()*20- len(currentGameState.getCapsules())*1000

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
