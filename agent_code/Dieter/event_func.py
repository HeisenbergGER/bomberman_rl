
import numpy as np 

def in_bomb_radius(game_state: dict):
    reamaining_time = 5
    agent_pos = game_state["self"][3]
    bomb_all = game_state["bombs"]
    field = game_state["field"]
    for bomb in bomb_all:
        bomb_pos = bomb[0]
        # there is a Bomb in Range aloing the y axis
        if(np.sum(np.abs(np.asarray(agent_pos) - np.asarray(bomb_pos))) <= 3 and (np.asarray(agent_pos) - np.asarray(bomb_pos))[0] == 0):
            reamaining_time = bomb[1]
        #Check if there is a wall inbetween 
            for field_prop in field[np.asarray(agent_pos)[0],np.arange(bomb_pos[1],agent_pos[1])]:
                if field_prop == -1:
                    reamaining_time = 5
                    break
        # there is a Bomb in Range aloing the X axis
        elif(np.sum(np.abs(np.asarray(agent_pos) - np.asarray(bomb_pos))) <= 3 and (np.asarray(agent_pos) - np.asarray(bomb_pos))[1] == 0):
            reamaining_time = bomb[1]
        #Check if there is a wall inbetween 
            for field_prop in field[np.arange(bomb_pos[0],agent_pos[0]),np.asarray(agent_pos)[1]]:
                if field_prop == -1:
                    reamaining_time = 5
                    break
    #returns the remaining time before the explosion, a value of 5 indicates no danger
    return reamaining_time



# checks that if in a 1 block radius around the agent there is a empty location that is not affected by a bomb if one would be dropped at the current agents location 
def check_for_cover(game_state: dict):
    agent_pos = game_state["self"][3]
    field = game_state["field"]
    if (field[np.asarray(agent_pos)[0]+1,np.asarray(agent_pos)[1]+1] == 0 or field[np.asarray(agent_pos)[0]+1,np.asarray(agent_pos)[1]-1] == 0
        or field[np.asarray(agent_pos)[0]-1,np.asarray(agent_pos)[1]+1] == 0 or field[np.asarray(agent_pos)[0]-1,np.asarray(agent_pos)[1]-1] == 0):
        return True
    else:
        return False  
        
def distance(a,b):
    return np.sqrt(np.abs(a[0]-b[0])**2 + np.abs(a[1] - b[1])**2)
        
def distance_to_enemy(game_state: dict):
    agent_pos = game_state["self"][3]
    enemy_names = []
    distances = []
    enemies = game_state["others"]
    for enemy in enemies:
        enemy_names.append(enemy[0])
        distances.append(distance(agent_pos, enemy[3])) 
    #closest = [x for _, x in sorted(zip(distances, enemy_names))][0]
    if distances != []:
        return np.min(distances)
    else: 
        return None
        
def distance_to_nearest_coin(game_state: dict):
    agent_pos = game_state["self"][3]
    coins_pos = game_state["coins"]
    distances = []
    for coin in coins_pos:
        distances.append(distance(coin,agent_pos))
    return np.min(distances)
    