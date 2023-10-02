from collections import namedtuple, deque
import pickle
from typing import List
import settings
import torch
import events as e
import numpy as np
from .callbacks import state_to_features, ACTIONS, device
from .event_func import in_bomb_radius, check_for_cover, distance_to_enemy, distance_to_nearest_coin
import matplotlib.pyplot as plt
import os 
import random
from environment import BombeRLeWorld 
import main

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'future_state'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = int(1024)  # keep only ... last transitions
EXPLORATION_RATE = 1.0
ROUND = 1
STEP = 1
BATCH_SIZE = 16

# Events
WITHIN_BLASTZONE = "WITHIN_BLASTZONE"
OUTSIDE_BLASTZONE = "OUTSIDE_BLASTZONE"
WALK_IN_BLAST = "WALK_IN_BLAST"
WALK_OUT_BLAST = "WALK_OUT_BLAST"
BAD_BOMB_PlACE = "BAD_BOMB_PlACE"
BOMB_DROPPED_IN_GOOD_PLACE = "BOMB_DROPPED_IN_GOOD_PLACE"
CLOSER_TO_ENEMY = "CLOSER_TO_ENEMY"
SAME_POSITION = "SAME_POSITION"
LOOP = "LOOP"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_TO_COIN = "FURTHER_TO_COIN"
ALL_COINS_COLLECTED = "ALL_COINS_COLLECTED"
TIME_STEP = "TIME_STEP"
NEW_FIELD = "NEW_FIELD"

TRAINING_INTERVAL = 1

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def add_transition(self, state, action, reward, future_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(state, action, reward, future_state)
        self.position = (self.position + 1) % self.capacity
    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)
        #return sample_sequences_fixed_length(self.buffer, 4, batch_size)
        
    def flush(self):
        self.buffer = []
        self.position = 0
    def __len__(self):
        return len(self.buffer)




def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')

    #remove old lists of analysis
    try:
        os.remove("Dieter_training_actions.txt")
        os.remove("Dieter_training_rewards.txt") 
        os.remove("Dieter_training_Q.txt")
        os.remove("huber_loss.txt")
    except:
        pass


    # set up lists to keep track of states and actions
    self.exploration_rate = EXPLORATION_RATE

    
    self.replay_buffer = ReplayBuffer(capacity=TRANSITION_HISTORY_SIZE)

    self.EPS_MIN = 0.05
    self.EPS_DEC = 0.99
    #settings.SCENARIOS["coin-heaven"]["COIN_COUNT"] = 5
    self.round_pos = []



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
    self.round_pos.append(pos_old)
    
    events.append(TIME_STEP)
    self.logger.debug(f'Add game event {TIME_STEP} in step {new_game_state["step"]}')
    if pos_current not in self.round_pos:
        events.append(NEW_FIELD)
        self.logger.debug(f'Add game event {NEW_FIELD} in step {new_game_state["step"]}')
    if (in_bomb_radius(old_game_state) - in_bomb_radius(new_game_state)) > 0:
        events.append(WALK_IN_BLAST)
        self.logger.debug(f'Add game event {WALK_IN_BLAST} in step {new_game_state["step"]}')
    if (in_bomb_radius(new_game_state) - in_bomb_radius(old_game_state)) < 0:
        events.append(WALK_OUT_BLAST)
        self.logger.debug(f'Add game event {WALK_OUT_BLAST} in step {new_game_state["step"]}')    
    if (check_for_cover(new_game_state) == False):
        events.append(BAD_BOMB_PlACE)
        self.logger.debug(f'Add game event {BAD_BOMB_PlACE} in step {new_game_state["step"]}')
    if(True):
        if ((check_for_cover(old_game_state) == True) and (self_action == 'BOMB')):
            events.append(BOMB_DROPPED_IN_GOOD_PLACE)
            self.logger.debug(f'Add game event {BOMB_DROPPED_IN_GOOD_PLACE} in step {new_game_state["step"]}')
    if ((distance_to_enemy(old_game_state) != None) and (distance_to_enemy(new_game_state) != None)):
        if(distance_to_enemy(old_game_state) > distance_to_enemy(new_game_state)):
            events.append(CLOSER_TO_ENEMY)
            self.logger.debug(f'Add game event {CLOSER_TO_ENEMY} in step {new_game_state["step"]}')
            
    if pos_current == pos_old:
        events.append(SAME_POSITION)
        self.logger.debug(f'Add game event {SAME_POSITION} in step {new_game_state["step"]}')
    if self.replay_buffer.position > 6:
        if (((self.replay_buffer.buffer[self.replay_buffer.position -1][0]["self"][3] == self.replay_buffer.buffer[self.replay_buffer.position -3][0]["self"][3])) and ((self.replay_buffer.buffer[self.replay_buffer.position -3][0]["self"][3] == self.replay_buffer.buffer[self.replay_buffer.position -5][0]["self"][3]))):
            events.append(LOOP)
            self.logger.debug(f'Add game event {LOOP} in step {new_game_state["step"]}')
    if (len(new_game_state["coins"]) > 0) and (len(old_game_state["coins"]) > 0):
        if distance_to_nearest_coin(new_game_state) < distance_to_nearest_coin(old_game_state):
             events.append(CLOSER_TO_COIN)
             self.logger.debug(f'Add game event {CLOSER_TO_COIN} in step {new_game_state["step"]}')
             
        if distance_to_nearest_coin(new_game_state) > distance_to_nearest_coin(old_game_state):
             events.append(FURTHER_TO_COIN)
             self.logger.debug(f'Add game event {FURTHER_TO_COIN} in step {new_game_state["step"]}')
             
    if len(old_game_state["coins"]) > 0 and len(new_game_state["coins"]) == 0:  
        events.append(ALL_COINS_COLLECTED)    
        self.logger.debug(f'Add game event {ALL_COINS_COLLECTED} in step {new_game_state["step"]}')
    self.replay_buffer.add_transition(old_game_state,self_action,reward_from_events(self, events), new_game_state)
    # state_to_features is defined in callbacks.py    

    if (len(self.replay_buffer) >= BATCH_SIZE):
        pass
        #print("training...")
        train(self)
        log(self)
    
def clip_rewards(rewards, min_value, max_value):
    return [max(min(reward, max_value), min_value) for reward in rewards]
    
def sample_sequences_fixed_length(input_list, seq, batch_size):
    len_input = len(input_list)
    inter = []
    while (len(inter) < batch_size):
        random_index = random.randint(seq,len_input-seq)
        inter.append(input_list[random_index-1])
        inter.append(input_list[random_index])
        inter.append(input_list[random_index+1])
    return inter
    

  
def train(self,end=False):
    if BATCH_SIZE > len(self.replay_buffer):
        size = len(self.replay_buffer)
    else:
        size = BATCH_SIZE
    self.model.train()
    if end:
        batch = self.replay_buffer.sample_batch(1)
    else:
        batch = self.replay_buffer.sample_batch(size)
    batch_size = len(batch)
    #self.replay_buffer.flush()
    self.train_states = torch.tensor(np.array([state_to_features(transition.state) for transition in batch])).float()
    self.train_future_states = torch.tensor(np.array([state_to_features(transition.future_state) for transition in batch])).float()
    self.train_rewards = torch.tensor(np.array([transition.reward for transition in batch])).float()
    self.train_actions = [transition.action for transition in batch]
    self.action_indices = torch.zeros((self.train_states.shape[0], len(ACTIONS))).bool()
    for i in range(len(self.train_actions)):
        self.action_indices[i][ACTIONS.index(self.train_actions[i])] = True
    
    self.train_states = self.train_states.to(device)
    self.train_future_states = self.train_states.to(device)
    self.train_rewards = self.train_rewards.to(device)
    self.action_indices = self.action_indices.to(device)
    
    self.q_states = self.model.forward(self.train_states)
    self.q_states_max = self.q_states[self.action_indices]
    self.q_future_states = self.model.forward(self.train_future_states)
    self.q_future_states_max = torch.max(self.q_future_states, dim=1)[0]
    self.q_target = self.train_rewards + self.model.gamma * self.q_future_states_max
    
    self.loss = self.model.loss(self.q_states_max, self.q_target)
    self.model.optimizer.zero_grad()
    self.loss.backward()
    self.model.optimizer.step()
    
def log(self):
        file = open("Dieter_training_rewards.txt", "a")
        file.write(str(sum(np.array(self.train_rewards.cpu()))))
        file.write("\n")
        file.close()
        
        file = open("Dieter_training_Q.txt", "a")
        q = self.q_states_max.cpu()
        file.write(str(sum(np.array(q.detach().numpy()))))
        file.write("\n")
        file.close()

        file_2 = open("Dieter_training_actions.txt", "a")
        file_2.write('\n'.join(self.train_actions))
        file_2.write("\n")
        file_2.close()

        file_3 = open("huber_loss.txt", "a")
        file_3.write(str(self.loss.item()))
        file_3.write("\n")
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
    #self.replay_buffer.add_transition(last_game_state,last_action,reward_from_events(self, events),last_game_state)
    #train(self,end=True)
    #log(self)
    #if len(self.replay_buffer) >= BATCH_SIZE:
        #print("training...")
    #train(self)
    #log(self)
    #self.replay_buffer.flush()
    self.exploration_rate = self.exploration_rate * self.EPS_DEC if self.exploration_rate > self.EPS_MIN else self.EPS_MIN
    # Store the model
    global ROUND
    #log(self)
    if (ROUND % 100)  == 0:
        #print("saving...")
        #log(self)
        #if settings.SCENARIOS["coin-heaven"]["COIN_COUNT"] < 50:
            #settings.SCENARIOS["coin-heaven"]["COIN_COUNT"] += 5
        with open("Dieter.pt", "wb") as file:
            pickle.dump(self.model, file)
    ROUND += 1
    self.round_pos = []


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        #ALL_COINS_COLLECTED: 100,
        e.COIN_FOUND: 5,
        #e.BOMB_DROPPED: -1,
        e.KILLED_OPPONENT: 100,
        e.CRATE_DESTROYED: 10,
        e.KILLED_SELF: -50,
        TIME_STEP: -1,
        e.GOT_KILLED: 0,
        #e.MOVED_UP: -1,
        #e.MOVED_DOWN: -1,
        #e.MOVED_LEFT: -1,
        #e.MOVED_RIGHT: -1,
        e.WAITED: -5,
        e.INVALID_ACTION: -5,
        NEW_FIELD: 5,
        #CLOSER_TO_COIN: 3,
        #FURTHER_TO_COIN: 0,
        LOOP: -5,
        WALK_IN_BLAST: -6,
        WALK_OUT_BLAST: 2,
        BOMB_DROPPED_IN_GOOD_PLACE: 10,
        #CLOSER_TO_ENEMY: 5,
        #SAME_POSITION: -10,
        BAD_BOMB_PlACE: -10,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    #reward_sum -= 1 # penalty for every step
    return reward_sum
