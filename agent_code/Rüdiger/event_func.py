
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


def tile_in_bomb_radius(tile_x,tile_y,game_state: dict):
    reamaining_time = 5
    tile = (tile_x,tile_y)
    bomb_all = game_state["bombs"]
    field = game_state["field"]
    for bomb in bomb_all:
        bomb_pos = bomb[0]
        # there is a Bomb in Range aloing the y axis
        if(np.sum(np.abs(np.asarray(tile) - np.asarray(bomb_pos))) <= 3 and (np.asarray(tile) - np.asarray(bomb_pos))[0] == 0):
            reamaining_time = bomb[1]
        #Check if there is a wall inbetween 
            for field_prop in field[np.asarray(tile)[0],np.arange(bomb_pos[1],tile[1])]:
                if field_prop == -1:
                    reamaining_time = 5
                    break
        # there is a Bomb in Range aloing the X axis
        elif(np.sum(np.abs(np.asarray(tile) - np.asarray(bomb_pos))) <= 3 and (np.asarray(tile) - np.asarray(bomb_pos))[1] == 0):
            reamaining_time = bomb[1]
        #Check if there is a wall inbetween 
            for field_prop in field[np.arange(bomb_pos[0],tile[0]),np.asarray(tile)[1]]:
                if field_prop == -1:
                    reamaining_time = 5
                    break
    #returns the remaining time before the explosion, a value of 5 indicates no danger
    return reamaining_time



# checks that if in a 1 block radius around the agent there is a empty location that is not affected by a bomb if one would be dropped at the current agents location 
def check_for_cover(game_state: dict):
    agent_pos = game_state["self"][3]
    field = game_state["field"]
    if ((field[np.asarray(agent_pos)[0],np.asarray(agent_pos)[1]+1] == 0 and field[np.asarray(agent_pos)[0]+1,np.asarray(agent_pos)[1]+1] == 0) or 
        (field[np.asarray(agent_pos)[0],np.asarray(agent_pos)[1]+1] == 0 and field[np.asarray(agent_pos)[0]-1,np.asarray(agent_pos)[1]+1] == 0) or 
        (field[np.asarray(agent_pos)[0],np.asarray(agent_pos)[1]-1] == 0 and field[np.asarray(agent_pos)[0]+1,np.asarray(agent_pos)[1]-1] == 0) or 
        (field[np.asarray(agent_pos)[0],np.asarray(agent_pos)[1]-1] == 0 and field[np.asarray(agent_pos)[0]-1,np.asarray(agent_pos)[1]-1] == 0) or
        (field[np.asarray(agent_pos)[0]+1,np.asarray(agent_pos)[1]] == 0 and field[np.asarray(agent_pos)[0]+1,np.asarray(agent_pos)[1]+1] == 0) or 
        (field[np.asarray(agent_pos)[0]+1,np.asarray(agent_pos)[1]] == 0 and field[np.asarray(agent_pos)[0]+1,np.asarray(agent_pos)[1]-1] == 0) or 
        (field[np.asarray(agent_pos)[0]-1,np.asarray(agent_pos)[1]] == 0 and field[np.asarray(agent_pos)[0]-1,np.asarray(agent_pos)[1]+1] == 0) or 
        (field[np.asarray(agent_pos)[0]-1,np.asarray(agent_pos)[1]] == 0 and field[np.asarray(agent_pos)[0]-1,np.asarray(agent_pos)[1]-1] == 0)
        ):
        return True
    else:
        return False  





def has_save_neigbour(tile_x,tile_y,game_state: dict):
    save = False
    neigbours = [(tile_x+1, tile_y),(tile_x-1, tile_y),(tile_x, tile_y+1),(tile_x, tile_y-1)]
    for tile in neigbours:
        if game_state["field"][tile[1],tile[0]] == 0:
            if (tile_in_bomb_radius(tile[0],tile[1],game_state) == 5):
                save = True
                break 
    return save

def distance_to_next_target(tile_x,tile_y,game_state: dict):
    targets_of_intrest = []

    for agents in game_state["others"]:
        targets_of_intrest.append(np.array(agents[3]))
    for coins in game_state["coins"]:
        targets_of_intrest.append(np.array(coins))
    crates = np.array(np.where(game_state["field"] == 1)).T
    for rows in crates:
        targets_of_intrest.append(rows)
    if targets_of_intrest:
        distance_to_next = np.min(np.linalg.norm(targets_of_intrest - np.array([tile_x,tile_y]), axis= 1))
        return distance_to_next
    else:
        return False
    

def distance_to_safety(tile_x,tile_y,game_state: dict):
    targets_of_intrest = []

    empty = np.array(np.where(game_state["field"] == 0)).T
    for rows in empty:
        if(tile_in_bomb_radius(*rows,game_state) == 5):
            targets_of_intrest.append(rows)
    if targets_of_intrest:
        distance_to_next = np.min(np.linalg.norm(targets_of_intrest - np.array([tile_x,tile_y]), axis= 1))
        return distance_to_next
    else:
        return False
    
