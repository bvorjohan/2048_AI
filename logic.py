#

from random import *
import numpy as np


def new_game(n):
    #####
    # Generate new matrix
    #####
    matrix = []

    for i in range(n):
        matrix.append([0] * n)
    return matrix

def num_zeros(mat):
    #####
    # Returns the number of 0s in a 4x4 matrix
    #####
    return 16-np.count_nonzero(mat)



def add_two(mat):
    #####
    # Adds a random tile
    #####
    a = randint(0, len(mat)-1)
    b = randint(0, len(mat)-1)
    while mat[a][b] != 0:
        a = randint(0, len(mat)-1)
        b = randint(0, len(mat)-1)
    mat[a][b] = (2 if random() < .9 else 4)
    return mat



def game_score(mat):
    #####
    # Evaluates the value of a game by taking the value of the greatest tile
    #####
    max_tile = 0
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] > max_tile:
                max_tile = mat[i][j]
    return max_tile

def game_state(mat):
    #####
    # Determines the state of the game
    #####    
    for i in range(len(mat)-1):
        for j in range(len(mat[0])-1):
            if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                return 'not over'
    if num_zeros(mat) > 0:
        return 'not over'

    for k in range(len(mat)-1):
        if mat[len(mat)-1][k] == mat[len(mat)-1][k+1]:
            return 'not over'
    for j in range(len(mat)-1):
        if mat[j][len(mat)-1] == mat[j+1][len(mat)-1]:
            return 'not over'
    return 'lose'



def reverse(mat):
    #####
    # Flips matrix laterally
    #####
    return np.flip(mat, 1)



def transpose(mat):
    #####
    # Transposes matrix
    #####
    return np.transpose(mat)




def cover_up(mat):
    #####
    # Performs a "swipe" without merging
    #####
    new = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    done = False
    for i in range(4):
        count = 0
        for j in range(4):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return new, done


def merge(mat):
    #####
    # Performs a merge operation - to be done after a swipe in "cover_up"
    #####
    done = False
    for i in range(4):
        for j in range(3):
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j] *= 2
                mat[i][j+1] = 0
                done = True
    return mat, done


def merge_score(mat):
    #####
    # Determines the score gained from a move (as in the original game)
    #####
    score = 0
    mult = 1
    done = False
    for i in range(4):
        for j in range(3):
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j]*=2
                mat[i][j+1]=0
                if i == 0 or i == 3:
                    mult+=1
                if j == 0 or j == 3:
                    mult+=1
                score += mat[i][j]*1
                done = True
    return mat, done, score

def up(game):
        # print("up")
        # return matrix after shifting up
        game = transpose(game)
        game, done = cover_up(game)
        temp = merge(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        game = transpose(game)
        return game, done

def down(game):
        # print("down")
        game = reverse(transpose(game))
        game, done=cover_up(game)
        temp = merge(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        game = transpose(reverse(game))
        return game, done

def left(game):
        # print("left")
        # return matrix after shifting left
        game, done = cover_up(game)
        temp = merge(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        return game, done

def right(game):
        # print("right")
        # return matrix after shifting right
        game = reverse(game)
        game, done = cover_up(game)
        temp = merge(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        game = reverse(game)
        return game, done


def up_score(game):
        # print("up")
        # return matrix and score after shifting up
        game = transpose(game)
        game, done = cover_up(game)
        temp = merge_score(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        game = transpose(game)
        score = temp[2]
        return game, done, score

def down_score(game):
        # print("down")
        # return matrix and score after shifting down
        game = reverse(transpose(game))
        game, done = cover_up(game)
        temp = merge_score(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        score = temp[2]
        game = transpose(reverse(game))
        return game, done, score

def left_score(game):
        # print("left")
        # return matrix and score after shifting left
        game, done = cover_up(game)
        temp = merge_score(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        score = temp[2]
        return game, done, score

def right_score(game):
        # print("right")
        # return matrix and score after shifting right
        game = reverse(game)
        game, done = cover_up(game)
        temp = merge_score(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        score = temp[2]
        game = reverse(game)
        return game, done, score


def brad_state_score(mat):
    #####
    # An alternative scoring function, one which tries to imitate an intuative, human style
    #####
    coord_list = [(0,0),(0,1),(0,2),(0,3),
                  (1, 3), (1, 2), (1, 1), (1, 0),
                  (2, 0), (2, 1), (2, 2), (2, 3),
                  (3, 3), (3, 2), (3, 1), (3, 0),]


    weight_mat = np.array([32768,16384,8192,4096, 2048, 1024, 512, 256, 128,64,32,16, 8, 4, 2, 1])

    biggest_tile = np.argmax([mat[0][0], mat[0][3], mat[3][0], mat[3][3]])

    score_0 = 0
    score_1 = 0

    tile_max_0 = [mat[0][0], mat[0][3], mat[3][0], mat[3][3]][biggest_tile]
    tile_max_1 = tile_max_0

    # biggest_tile = 0

    if biggest_tile == 0:
        for i, coord in enumerate(coord_list):
            # print(coord)
            # x=coord[0]
            # y=coord[1]
            # print(x)
            # print(y)
            tile_0 = mat[coord[0]][coord[1]]
            tile_1 = mat[coord[1]][coord[0]]
            if tile_0 <= tile_max_0:
                score_0 += tile_0 * weight_mat[i]
                tile_max_0 = tile_0
            if tile_1 <= tile_max_1:
                score_1 += tile_1 * weight_mat[i]
                tile_max_1 = tile_1
        return np.amax([score_0,score_1])
    elif biggest_tile == 1:
        for i, coord in enumerate(coord_list):
            tile_0 = mat[coord[0]][3 - coord[1]]
            tile_1 = mat[3 - coord[1]][coord[0]]
            if tile_0 <= tile_max_0:
                score_0 += tile_0 * weight_mat[i]
                tile_max_0 = tile_0
            if tile_1 <= tile_max_1:
                score_1 += tile_1 * weight_mat[i]
                tile_max_1 = tile_1
        return np.amax([score_0,score_1])
    elif biggest_tile == 2:
        for i, coord in enumerate(coord_list):
            tile_0 = mat[3 - coord[0]][coord[1]]
            tile_1 = mat[coord[1]][3 - coord[0]]
            if tile_0 <= tile_max_0:
                score_0 += tile_0 * weight_mat[i]
                tile_max_0 = tile_0
            if tile_1 <= tile_max_1:
                score_1 += tile_1 * weight_mat[i]
                tile_max_1 = tile_1
        return np.amax([score_0,score_1])
    elif biggest_tile == 3:
        for i, coord in enumerate(coord_list):
            tile_0 = mat[3 - coord[0]][3 - coord[1]]
            tile_1 = mat[3 - coord[1]][3 - coord[0]]
            if tile_0 <= tile_max_0:
                score_0 += tile_0 * weight_mat[i]
                tile_max_0 = tile_0
            if tile_1 <= tile_max_1:
                score_1 += tile_1 * weight_mat[i]
                tile_max_1 = tile_1
        return np.amax([score_0, score_1])





def zeros_to_steps(zeros):
    #####
    # Converts the number of zeros on the board to iterative steps
    #####
    if zeros < 2:
        return 4
    elif zeros < 6:
        return 3
    else:
        return 2
