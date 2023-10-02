import os
import pickle
import random
import numpy as np
from agent_code.Rüdiger.model import REG_FOR
from .event_func import in_bomb_radius, tile_in_bomb_radius, distance_to_next_target, check_for_cover, has_save_neigbour, distance_to_safety

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When ina training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("Ruediger.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = REG_FOR()
    else:
        self.logger.info("Loading model from saved state.")
        with open("Ruediger.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    if self.train and random.random() < self.exploration_rate:
        self.logger.debug("Choosing action purely at random.")
        self.logger.debug(f'Exploration rate; {self.exploration_rate}')
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    features = state_to_features(game_state)


    if np.shape(np.shape(features))[0] == 1:
        features = features.reshape(1,-1)

    
    q_pred = self.model.pred(np.concatenate((features.reshape(1,-1), np.zeros(len(features)).reshape((-1,1))), axis=1))
    for i in [1,2,3,4]:
        q_pred = np.concatenate((q_pred,self.model.pred(np.concatenate((features, np.ones(len(features)).reshape((-1,1))*i), axis=1))), axis=0)

    action = q_pred.argmax(axis=0)
    


    self.logger.info(f'Doing Action: {action}')
    return ACTIONS[action]

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends


    #only the tiles around agent ruediger are considered now
    """The tile of agent ruediger itself is the 0th entry of the feature vector and can take values depending on if the tile is in danger by a bomb
    explosion, or if a bomb is dropped there there is cover available and a way to get there. 
    from there on the tiles are numerd from bottom left to top right leaving out the middle tile(ruedigers tile).
    like this: 
                              4
                            2 0 3
                              1  
    Except the 0th tile, tiles can take values beteween 0 and 6 where:
    0 = coin
    1 = other player 
    2 = crate 
    3 = empty and closest to next target
    4 = empty 
    5 = in danger but close to safety
    6 = wall
    7 = in danger of a bomb explosion
    """     
    
    if game_state is None:
        return None

    vector = np.zeros(5)
    distances = np.zeros(5)
    distances[0] = 100
    distance_safety = np.zeros(5)
    distance_safety[0] = 100
    tile_rel_coord = np.array([[0,1],
                              [-1,0], [1,0],
                              [0,-1]])

    for i,elem in enumerate(vector):
        #Player in Danger?
        if i == 0:
            if (in_bomb_radius(game_state) != 5):
                vector[i] = 0
                continue
            elif (check_for_cover(game_state) == True):
                vector[i] = 2
                continue    
            else: 
                vector[i] = 1
                continue


        tile_coord = np.asarray(game_state["self"][3]) + tile_rel_coord[i-1]
        distances[i] = distance_to_next_target(*tile_coord, game_state)
        distance_safety[i] = distance_to_safety(*tile_coord, game_state)

        if game_state['field'][tile_coord[0],tile_coord[1]] == -1:
            vector[i] = 6
            continue
        elif game_state['field'][tile_coord[0],tile_coord[1]] == 1:
            vector[i] = 2
            continue
        elif(any(tuple(tile_coord) in sublist for sublist in game_state["others"])):
            vector[i] = 1
            continue
        elif(tile_in_bomb_radius(*tile_coord, game_state) < 5 or game_state['explosion_map'][tile_coord[0],tile_coord[1]] != 0):
            vector[i] = 7
            continue
        elif(tuple(tile_coord) in game_state["coins"]):
            vector[i] = 0
            continue
        else:
            vector[i] = 4



    if (4 in vector and (2 not in vector[1:]) and (1 not in vector[1:]) and (0 not in vector[1:])):
        tile_cloest_to_spot = np.argmin(np.where(vector==4,1,100)*distances)
        
        vector[tile_cloest_to_spot] = 3


    if (7 in vector):
        tile_cloest_to_safety = np.argmin(np.where(vector==7,1,100)*distance_safety)
        
        vector[tile_cloest_to_safety] = 5  


    return vector
