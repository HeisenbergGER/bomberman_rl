import os
import pickle
import random
import torch
import numpy as np
from agent_code.Dieter.model import DQN, CONV_DQN

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("Dieter.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = CONV_DQN(17,6).to(device)
    else:
        self.logger.info("Loading model from saved state.")
        with open("Dieter.pt", "rb") as file:
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
    with torch.no_grad():
        self.model.eval()
        pred_rewards = self.model.forward(torch.tensor(np.array(features)).float()).cpu()
    #print(pred_rewards)
    action = torch.argmax(pred_rewards)
    #print("Action: ",action.item())
    self.logger.info(f'Doing Action: {action}')
    #print("Sting: ",ACTIONS[action.item()])
    #print(ACTIONS[action.item()])
    return ACTIONS[action.item()]
    
def z_score_normalize(data):
    mean = np.mean(data, axis=(0, 1, 2))  # Calculate mean across all channels and positions
    std = np.std(data, axis=(0, 1, 2))    # Calculate standard deviation across all channels and positions
    normalized_data = (data - mean) / std
    return normalized_data

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
    full_grid = True
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    grid_shape = game_state["field"].shape
    if full_grid:
        vector = np.zeros((6,grid_shape[0],grid_shape[1]))
        for instance in game_state["others"]:
            pos = instance[3]
            #enemy position in channel 0
            vector[0,pos[0],pos[1]] = 1
    
        for instance in game_state["bombs"]:
            pos = instance[0]
            #bomb position in channel 1
            vector[1,pos[0],pos[1]] = 1

        for instance in game_state["coins"]:
            pos = instance
            #coin position in channel 2
            vector[2,pos[0],pos[1]] = 1

        #crate position in channel 3
        vector[3,:,:] += game_state["field"]
        #wall position in channel 4
        vector[4,:,:] -= game_state["field"]
        #own position in channel 5
        own_pos = game_state["self"][3]
        vector[5,own_pos[0],own_pos[1]] = 1
        
        #vector = vector.reshape((6,17,17))
        #return z_score_normalize(vector)
        #print(vector.flatten().shape)
        return vector
    else:
        vector = np.zeros((9,9,5))

