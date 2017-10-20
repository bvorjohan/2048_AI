from tkinter import *
from logic import *
from random import *
from time import *
import numpy as np
from copy import deepcopy
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

LR = 1e-3

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
                            32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61",
                            512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"}
CELL_COLOR_DICT = {2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
                    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
                    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2"}
FONT = ("Verdana", 40, "bold")

MAX_MOVES = 4096

# ACTION_LIST = [KEY_UP,KEY_DOWN,KEY_LEFT,KEY_RIGHT,]

KEY_UP_ALT = "\'\\uf700\'"
KEY_DOWN_ALT = "\'\\uf701\'"
KEY_LEFT_ALT = "\'\\uf702\'"
KEY_RIGHT_ALT = "\'\\uf703\'"

KEY_UP = "'w'"
KEY_DOWN = "'s'"
KEY_LEFT = "'a'"
KEY_RIGHT = "'d'"

ACTION_LIST = [KEY_UP,KEY_DOWN,KEY_LEFT,KEY_RIGHT,]


class DQN_Manager:

    def __init__(self):
        self.memory = []
        self.model = self.neural_network_model()
        self.action_size = 4
        self.gamma = .95
        self.epsilon = 1.0
        self.epsilon_min = .01
        self.epsilon_decay = .95

    def neural_network_model(self,width=4,height=4):
        network = input_data(shape = [None, width, height, 1], name = "input")
        # network = input_data(placeholder = [None, width, height, 1], name = "input")


        network = conv_2d(network, 16, 2, padding = "same")
        metwork = max_pool_2d(network, 2, padding = "same")

        network = conv_2d(network, 16, 2, padding = "valid")
        metwork = max_pool_2d(network, 2, padding = "valid")

        network = fully_connected(network, 128, activation = "relu")
        network = dropout(network, 0.8)

        network = fully_connected(network, 64, activation = "relu")
        network = dropout(network, 0.8)

        network = fully_connected(network, 4, activation = "softmax")
        network =  regression(network, optimizer="adam", learning_rate=LR, loss="categorical_crossentropy",name="targets")

        model = tflearn.DNN(network, tensorboard_dir="log")
        return model

    def remember(self,state,action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # print(np.array(state).reshape(4,4,1))
        # action_list = [KEY_UP,KEY_DOWN,KEY_LEFT,KEY_RIGHT,]
        if np.random.rand() <= self.epsilon:
            return randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):

        minibatch = sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, n_epoch=1, snapshot_step=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print(self.epsilon)

        """
        fit_data = [[],[]]
        for state, action, reward, next_state, done in self.memory:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            fit_data[0].append(state[0])
            fit_data[1].append(target_f)
        self.model.fit(fit_data[0], fit_data[1])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print(self.epsilon)
        """

class Tree_Search_Manager:
    def __init__(self):
        self.commands = {   KEY_UP: up_score, KEY_DOWN: down_score, KEY_LEFT: left_score, KEY_RIGHT: right_score,
                            KEY_UP_ALT: up_score, KEY_DOWN_ALT: down_score, KEY_LEFT_ALT: left_score, KEY_RIGHT_ALT: right_score }
        self.commands_2 = {   KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left, KEY_RIGHT: right,
                            KEY_UP_ALT: up, KEY_DOWN_ALT: down, KEY_LEFT_ALT: left, KEY_RIGHT_ALT: right }

    def make_copy(thing):
        return deepcopy(thing)

    def evaluate_n_steps_max(self,num_steps,game_state):
        def recursive_eval(state_matrix, current_depth):
            possible_states = []
            # print("state matrix: ", state_matrix)
            for row_num, row in enumerate(state_matrix):
                for tile_num, tile in enumerate(row):
                    # print(row)
                    if tile == 0:
                            for new_tile in [2,4]:
                                # print(state_matrix)
                                possible_states.append(deepcopy(state_matrix))
                                possible_states[-1][row_num][tile_num] = new_tile
                                # print("possible state: ",possible_states[-1])
            if len(possible_states) == 0:
                possible_states.append(state_matrix)
            action_score_list = []
            count=0
            for state in possible_states:
                for action in ACTION_LIST:
                    # print(state)
                    # _ = input("")
                    new_matrix,_,reward = self.commands[action](deepcopy(state))
                    # print(action)
                    # print(reward)
                    # print(state)
                    # print(new_matrix)
                    # _ = input("...")
                    # /do move on matrix without generating new tiles -> new_matrix
                    # /find score from doing that move -> reward
                    # /do not evaluate impossible moves
                    if new_matrix != game_state:
                        action_score_list.append([new_matrix, reward])
            # print(len(action_score_list))
            if current_depth == num_steps:
                max_list = [i[1] for i in action_score_list]
                return max(max_list)

            else:
                for action_score_data_num, action_score_data in enumerate(action_score_list):
                    action_score_list[action_score_data_num][1] = recursive_eval(deepcopy(action_score_data[0]), current_depth+1)
                max_list = [i[1] for i in action_score_list]
                # print("This is the return", max(max_list))
                return max(max_list)
        #----
        action_score_list = []

        for action in ACTION_LIST:
            new_matrix,_,reward = self.commands[action](deepcopy(game_state))
            # print(action)
            # print(reward)
            # print(new_matrix)
            # /do move on matrix without generating new tiles -> new_matrix
            # /find score from doing that move -> reward
            # /do not evaluate impossible moves
            if new_matrix != game_state:
                action_score_list.append([action, new_matrix, reward])
        if ((num_steps == 0) or (len(action_score_list) < 2 )):
            max_list = [i[2] for i in action_score_list]
            index = max_list.index(max(max_list))
            return action_score_list[index][0]
        else:
            score_list = []

            for action_score_data_num, action_score_data in enumerate(action_score_list):

                score_list.append(action_score_data[2]+0.9*recursive_eval(deepcopy(action_score_data[1]), 1))
                # action_score_list[action_score_data_num][2] += recursive_eval(action_score_data[1], 1)
            index = score_list.index(max(score_list))
            return action_score_list[index][0]

    def run_simple_metric_game(self):
        game = Game_Grid()
        for _ in range(MAX_MOVES):
            # sleep(1)
            action = self.evaluate_n_steps_max(zeroes_to_steps(num_zeroes(game.matrix)),game.matrix)
            print(action)
            print(game.matrix)
            game.matrix, done = game.commands[action](game.matrix)

            if done:
                game.matrix = add_two(game.matrix)
                game.update_grid_cells()
                done = False

            is_finished = True if game_state(game.matrix) == 'lose' else False

            if is_finished:
                break
        game.destroy()





class Multi_Game_Manager:

    def run_random_games(self,num_games):
        for game_num in range(num_games):
            current_game = Game_Grid()
            current_game.run_random_game()
            current_game.destroy()

    def collect_data_with_score_threshold(self,score_threshold,num_games):
        pass #ILLEGAL METHOD
        training_data = []
        scores = []
        accepted_scores = []
        for _ in range(num_games):
            game = Game_Grid()
            score = 0
            game_memory = []
            prev_observation = []
            for _ in range(MAX_MOVES):
                action = game.pick_random_move()
                sleep(.01)
                game.matrix,done = game.commands[action](game.matrix)

                if done:
                    game.matrix = add_two(game.matrix)
                    # game.update_grid_cells()
                    # score=game_score(self.matrix)
                    done=False

                observation = game.matrix

                if len(prev_observation)>0:
                    game_memory.append([prev_observation, action])

                prev_observation = observation

                if game_state(game.matrix)=='lose':
                    score = game.get_score()
                    break
            print(score)
            if score >= score_threshold:
                print("Accepted!")
                accepted_scores.append(score)
                for data in game_memory:
                    if data[1] == KEY_UP:
                        output = [1,0,0,0]
                    if data[1] == KEY_DOWN:
                        output = [0,1,0,0]
                    if data[1] == KEY_LEFT:
                        output = [0,0,1,0]
                    if data[1] == KEY_RIGHT:
                        output = [0,0,0,1]
                    training_data.append([data[0],output])

            game.destroy()
            scores.append(score)
        training_data_save = np.array(training_data)
        np.save("2048_training_data_10000_samples_256_threshold.npy", training_data_save)

    def neural_network_model(self,width=4,height=4):
        network = input_data(shape = [None, width, height], name = "input")

        network = conv_2d(network, 1, [2,2], padding = "same")
        metwork = max_pool_2d(network, [2,2], padding = "same")

        network = conv_2d(network, 1, [2,2], padding = "valid")
        metwork = max_pool_2d(network, [2,2], padding = "valid")

        network = fully_connected(network, 128, activation = "relu")
        network = dropout(network, 0.8)

        network = fully_connected(network, 64, activation = "relu")
        network = dropout(network, 0.8)

        network = fully_connected(network, 4, activation = "softmax")
        network =  regression(network, optimizer="adam", learning_rate=LR, loss="categorical_crossentropy",name="targets")

        model = tflearn.DNN(network, tensorboard_dir="log")
        return model

    def train_model_from_file(self,file_,is_model=False):
        training_data = np.load(file_)
        X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]),1)
        new_array = []
        for data_num in range(len(X)):
            new_array.append([])
            for row_num in range(len(X[data_num])):
                new_array[data_num].append(X[data_num][row_num][0])
        X = new_array
        y = [i[1] for i in training_data]

        model = self.neural_network_model(input_size = len(X[0]))

        model.fit({"input":X}, {"targets":y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id="2048 Training Data 1")
        return model

    def deep_Q_network_runner(self, model):
        pass
        def remember(action, reward, next_state, done):
            memory.append((state, action, reward, next_state, done))
        memory = []
        gamma = .95

        for state, action, reward, next_state, done in memory:
            if done:
                target =  reward
            else:
                target = reward + gamma * np.amax(model.predict(next_state)[0])

            target_f = model.predict(state)
            target_f[0][action] = target

            model.fit(state, target_f, n_epoch = 1)


class Game_Grid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title("2048 AI!!!")

        #self.master.bind(...)
        self.commands = {   KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left, KEY_RIGHT: right,
                            KEY_UP_ALT: up, KEY_DOWN_ALT: down, KEY_LEFT_ALT: left, KEY_RIGHT_ALT: right }

        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()
        self.pack()

        # self.mainloop()

    def get_score(self):
        return game_score(self.matrix)

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def init_matrix(self):
        self.matrix = new_game(4)

        self.matrix=add_two(self.matrix)
        self.matrix=add_two(self.matrix)

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number])
        self.update_idletasks()

    def pick_random_move(self):
        return choice([KEY_UP,KEY_DOWN,KEY_LEFT,KEY_RIGHT,])

    def get_state(self):
        return np.array(self.matrix).reshape([-1,4,4,1])

    def run_random_game(self):
        score=0
        for moves in range(MAX_MOVES):
            # sleep(.1)
            self.matrix,done = self.commands[self.pick_random_move()](self.matrix)
            if done:
                self.matrix = add_two(self.matrix)
                self.update_grid_cells()
                score=game_score(self.matrix)
                done=False
            if game_state(self.matrix)=='lose':
                break
        print(score)


def run_DQN():
    DQN = DQN_Manager()
    for game_num in range(100):
        game = Game_Grid()
        score = 0
        # game_memory = []
        prev_observation = []
        for _ in range(MAX_MOVES):
            # sleep(.05)
            current_state = game.get_state()
            action = DQN.act(game.get_state())
            game.matrix,done = game.commands[ACTION_LIST[action]](game.matrix)


            if done:
                game.matrix = add_two(game.matrix)
                # game.update_grid_cells()
                # score=game_score(self.matrix)
                done=False

            observation = game.get_state()
            reward = game.get_score() - score
            is_finished = True if game_state(game.matrix)=='lose' else False
            DQN.remember(current_state, action, reward, observation, is_finished)


            prev_observation = observation
            if game_state(game.matrix)=='lose':
                score = game.get_score()
                break
        print(score)
        print(game_num)
        sleep(game_num if game_num>90 else 1)
        game.destroy()
        DQN.replay(32)


if __name__ == "__main__":
    TSM = Tree_Search_Manager()
    # print(TSM.evaluate_n_steps_max(2,[[0,0,128,2,],[0,0,0,2,],[256,0,0,0,],[16,16,8,4,],]))
    TSM.run_simple_metric_game()
