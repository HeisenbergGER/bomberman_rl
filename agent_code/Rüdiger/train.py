from collections import namedtuple, deque

import pickle
from typing import List
import settings
import torch
import events as e
import numpy as np
from .callbacks import state_to_features, ACTIONS
from .event_func import in_bomb_radius, check_for_cover, distance_to_next_target, has_save_neigbour, distance_to_safety
import os 
import matplotlib.pyplot as plt
import random 


# Events
WITHIN_BLASTZONE = "WITHIN_BLASTZONE"
OUTSIDE_BLASTZONE = "OUTSIDE_BLASTZONE"
IN_BLAST = "IN_BLAST"
WALK_OUT_BLAST = "WALK_OUT_BLAST"
BOMB_DROPPED_IN_GOOD_PLACE = "BOMB_DROPPED_IN_GOOD_PLACE"
REPEAT_ACTION = "REPEAT_ACTION"
MOVED_CLOSER = "MOVED_CLOSER"
BAD_BOMB_PLACE = "BAD_BOMB_PLACE" 
MOVED_IN_SAFE_DIR = "MOVED_IN_SAFE_DIR"




#IN this version Agent ruedigers regression ofrest is Updated after every  turn



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """


    # set up lists to keep track of states and actions
    self.exploration_rate = 1
    self.EPS_MIN = 0.2
    self.EPS_DEC = 0.99
    self.train_states = []
    self.train_actions = []
    self.train_future_states = []
    self.train_rewards = []


    self.train_states_round = []
    self.train_actions_round = []
    self.train_future_states_round = []
    self.train_rewards_round = []

    self.loss = []
    
    try:
        os.remove("ruediger_training_actions.txt")
        os.remove("ruediger_training_rewards.txt") 
        os.remove("ruediger_training_Q.txt")
        os.remove("ruediger_oob.txt")
    except:
        pass

    

    #The Regression forest is initialized with some examples of good and bad moves



    _ = self.model.fit(np.array([np.array([5,4,4,5,5,2]),np.array([5,4,4,5,5,3]),
                                np.array([5,5,4,5,4,0]),np.array([5,5,4,5,4,3]),
                                np.array([5,5,5,4,4,1]),np.array([5,5,5,4,4,0]),
                                np.array([5,4,5,4,5,2]), np.array([5,4,5,4,5,1]),

                                np.array([5,4,4,5,5,4]),np.array([5,4,4,5,5,5]),
                                np.array([5,5,4,5,4,4]),np.array([5,5,4,5,4,5]),
                                np.array([5,5,5,4,4,4]),np.array([5,5,5,4,4,5]),
                                np.array([5,4,5,4,5,4]), np.array([5,4,5,4,5,5])                    
                                 ]),
                       
                       np.concatenate((np.ones(8)*0.05,np.ones(8)*(-0.05))))




def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    current_bombs = new_game_state["bombs"]
    pos_current = new_game_state["self"][3]
    pos_old = old_game_state["self"][3]


    if (state_to_features(new_game_state)[0] == 0 and self_action != "BOMB"):
        events.append(IN_BLAST)
        self.logger.debug(f'Add game event {IN_BLAST} in step {new_game_state["step"]}')
        if(distance_to_safety(*pos_current, new_game_state) >  distance_to_safety(*pos_old, old_game_state)):
            events.append(MOVED_IN_SAFE_DIR)
            self.logger.debug(f'Add game event {MOVED_IN_SAFE_DIR} in step {new_game_state["step"]}')
        
    if (state_to_features(new_game_state)[0] != 0 and state_to_features(old_game_state)[0] == 0):
        events.append(WALK_OUT_BLAST)
        self.logger.debug(f'Add game event {WALK_OUT_BLAST} in step {new_game_state["step"]}')    

    
    if(new_game_state["step"] >3):    
        if (ACTIONS.index(self_action) in self.train_actions[-3:]):
            events.append(REPEAT_ACTION)
            self.logger.debug(f'Add game event {REPEAT_ACTION} in step {new_game_state["step"]}')
    
    if(distance_to_next_target(*pos_old, old_game_state)>distance_to_next_target(*pos_current,new_game_state)):
            events.append(MOVED_CLOSER)
            self.logger.debug(f'Add game event {MOVED_CLOSER} in step {new_game_state["step"]}')


    if(state_to_features(old_game_state)[0] ==1 and self_action == "BOMB"):
        events.append(BAD_BOMB_PLACE)
        self.logger.debug(f'Add game event {BAD_BOMB_PLACE} in step {new_game_state["step"]}')
    elif(state_to_features(old_game_state)[0] == 2 and self_action == "BOMB"):
        events.append(BOMB_DROPPED_IN_GOOD_PLACE)
        self.logger.debug(f'Add game event {BOMB_DROPPED_IN_GOOD_PLACE} in step {new_game_state["step"]}')

    


    

    # state_to_features is defined in callbacks.py



    self.train_states_round.append(state_to_features(old_game_state))
    self.train_actions_round.append(self_action)
    self.train_future_states_round.append(state_to_features(new_game_state))
    self.train_rewards_round.append(reward_from_events(self, events))
    self.train_actions_round[-1] = ACTIONS.index(self.train_actions_round[-1])




def log(self,q_max,oob):
        file = open("ruediger_training_rewards.txt", "a")
        file.write(str(self.train_rewards_round))
        file.close()
        
        file = open("ruediger_training_Q.txt", "a")
        file.write(str(np.ndarray.tolist(q_max)))
        file.close()

        file_2 = open("ruediger_training_actions.txt", "a")
        file_2.write(str(self.train_actions_round))
        file_2.close()


        file_3 = open("ruediger_oob.txt", "a")
        file_3.write(str(oob))
        file_3.write(",")
        file_3.close()






def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    
    self.train_states_round.append(state_to_features(last_game_state))
    self.train_future_states_round.append(np.array([1,6,6,6,6]))
    self.train_actions_round.append(last_action)
    self.train_rewards_round.append(reward_from_events(self, events))
    self.train_actions_round[-1] = ACTIONS.index(self.train_actions_round[-1])

    
    

    self.train_states.extend(self.train_states_round)
    self.train_actions.extend(self.train_actions_round)
    self.train_future_states.extend(self.train_future_states_round)
    self.train_rewards.extend(self.train_rewards_round)



    if(len(self.train_states)%20 == 0):

        train_rewards = np.array(self.train_rewards)
        train_states = np.array(self.train_states)
        train_future_states = np.array(self.train_future_states)
        train_actions = np.array(self.train_actions)



        
        #Updated the Regression forest with the new rewards
        q_future_states_all = self.model.pred(np.concatenate((train_future_states, np.zeros(len(train_future_states)).reshape((-1,1))), axis=1))

        for i in [1,2,3,4]:
            q_future_states_all = np.column_stack((q_future_states_all,self.model.pred(np.concatenate((train_future_states, np.ones(len(train_future_states)).reshape((-1,1))*i), axis=1))))


        q_future_states_max = np.max(q_future_states_all, axis = 1)

        

        #calculate ground truth
        q_target = train_rewards + self.model.gamma * q_future_states_max
        

        _ = self.model.fit(np.concatenate((train_states, np.array(train_actions).reshape((-1,1))), axis=1), q_target)


        self.exploration_rate = self.exploration_rate * self.EPS_DEC if self.exploration_rate > self.EPS_MIN else self.EPS_MIN
        with open("Ruediger.pt", "wb") as file:
            pickle.dump(self.model, file)
        

        log(self,q_future_states_max, self.model.oob_score())


    self.train_states_round = []
    self.train_actions_round = []
    self.train_future_states_round = []
    self.train_rewards_round = []



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.CRATE_DESTROYED: 0.3,
        e.COIN_COLLECTED: 0.2,
        e.KILLED_OPPONENT: 1,
        BOMB_DROPPED_IN_GOOD_PLACE: 0.7,
        e.KILLED_SELF: -0.5,
        MOVED_CLOSER: 0.04,
        MOVED_IN_SAFE_DIR: 0.05,

        e.WAITED: -0.02,
        e.MOVED_LEFT: -0.01,
        e.MOVED_RIGHT: -0.01,
        e.MOVED_UP: -0.01, 
        e.MOVED_DOWN: -0.01, 
        e.INVALID_ACTION: -0.05,
   }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
