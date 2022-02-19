from player import Player
import numpy as np
from config import CONFIG
import copy, random

class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def add_gaussian_noise(self, player:Player, mean:float, standard_deviation:float):
        first_layer_weights = player.nn.first_layer_weights_matrix
        gaussian_noise = np.random.normal(mean,standard_deviation,first_layer_weights.shape)
        first_layer_weights += gaussian_noise
        second_layer_weights = player.nn.second_layer_weights_matrix
        gaussian_noise = np.random.normal(mean,standard_deviation,second_layer_weights.shape)
        second_layer_weights += gaussian_noise
        b1 = player.nn.b1
        gaussian_noise = np.random.normal(mean,standard_deviation,b1.shape)
        b1 += gaussian_noise
        b2 = player.nn.b2
        gaussian_noise = np.random.normal(mean,standard_deviation,b2.shape)
        b2 += gaussian_noise
        return player

    def mutate(self, child):

        # TODO
        # child: an object of class `Player`
        return self.add_gaussian_noise(child, 0.0,0.5)

    def q_tournement(self,players,  num_players, q):
        result = []
        for i in range(num_players):
            batch = []
            for j in range(q):
                batch.append(np.random.choice(players))
            result.append(copy.deepcopy(sorted(batch, key=lambda x: x.fitness, reverse=True)[0]))
        return result
        # chosen_list = []
        # for i in range(num_players):
        #     q_players = np.random.randint(0 ,len(players), q)
        #     chosen_list.append(max([players[x] for x in q_players]))
        # return chosen_list
    
    def crossover(self, parent_players, num_players):
        children = []
        w1_row_num = parent_players[0].nn.first_layer_weights_matrix.shape[0]
        w2_row_num = parent_players[0].nn.second_layer_weights_matrix.shape[0]
        b1_shape = parent_players[0].nn.b1.shape[0]
        b2_shape = parent_players[0].nn.b2.shape[0]
        index = 0
        for i in range(num_players//2):
            parent1 = parent_players[index]
            parent2 = parent_players[index+1]
            child1 = Player('helicopter')
            child2 = Player('helicopter')
            child1.nn.first_layer_weights_matrix = np.concatenate((parent1.nn.first_layer_weights_matrix[:w1_row_num//2], parent2.nn.first_layer_weights_matrix[w1_row_num//2:]), axis = 0)
            child2.nn.first_layer_weights_matrix = np.concatenate((parent2.nn.first_layer_weights_matrix[:w1_row_num//2], parent1.nn.first_layer_weights_matrix[w1_row_num//2:]), axis = 0)
            child2.nn.second_layer_weights_matrix = np.concatenate((parent1.nn.second_layer_weights_matrix[:w2_row_num//2], parent2.nn.second_layer_weights_matrix[w2_row_num//2:]), axis=0)
            child2.nn.second_layer_weights_matrix = np.concatenate((parent2.nn.second_layer_weights_matrix[:w2_row_num//2], parent1.nn.second_layer_weights_matrix[w2_row_num//2:]), axis = 0)
            # parent1.nn.first_layer_weights_matri = child1_w1
            # parent1.nn.second_layer_weights_matri = child1_w2
            # parent2.nn.first_layer_weights_matrix = child2_w1
            # parent2.nn.first_layer_weights_matrix = child2_w2
            children.extend([parent1, parent2])
            index += 2
        return children

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:

            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects
            # parents = self.roulette_wheel_selection(prev_players, num_players)
            # TODO (additional): a selection method other than `fitness proportionate`
            parents = self.q_tournement(prev_players,num_players,5)
            # TODO (additional): implementing crossover
            parents_copy = copy.deepcopy(parents)
            # parents_copy = self.crossover(parents_copy, num_players)
            # parents_copy = copy.deepcopy(parents)
            new_players = [self.mutate(x) for x in parents_copy]
            return new_players

    def roulette_wheel_selection(self,players,  num_players):
        population_fitness = sum([p.fitness for p in players])
        player_probabilities = [p.fitness/population_fitness for p in players]
        return list(np.random.choice(players,num_players, p=player_probabilities))

    def next_population_selection(self, players, num_players):

        # # TODO
        # # num_players example: 100
        # # players: an array of `Player` objects
        # return players[: num_players]
        # TODO (additional): a selection method other than `top-k`
        return self.roulette_wheel_selection(players, num_players)
        # TODO (additional): plotting

        

    